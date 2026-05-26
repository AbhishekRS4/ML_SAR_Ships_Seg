import os
import gc
import time
import torch
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist


from pathlib import Path
from torch import GradScaler
from torch.optim import SGD, AdamW
from typing import Union, List
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import PolynomialLR
from torch.nn.functional import softmax


from loss_func.focal_loss import FocalLoss
from data_handler.data_loader_dali import (
    build_dali_tfrecords_loader,
)
from models.sem_seg_model import (
    ResNet34UNet,
    PSAResNet34UNet,
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
)
from metrics.compute_metrics import (
    SemSegMetricsCalculator,
    get_confusion_matrix_figure,
)


def setup_ddp(rank: int, world_size: int) -> None:
    """
    initialize the distributed process group

    ---------
    Arguments
    ---------
    rank: int
        the rank of the current process
    world_size: int
        the total number of processes
    """
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    return


def cleanup_ddp() -> None:
    """
    destroy the distributed process group
    """
    dist.destroy_process_group()
    return


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    reduce a tensor across all processes by averaging

    ---------
    Arguments
    ---------
    tensor: torch.Tensor
        the tensor to reduce
    world_size: int
        the total number of processes

    -------
    Returns
    -------
    tensor: torch.Tensor
        the reduced tensor (averaged across all processes)
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_nn_ddp_dali(
    model: DDP,
    device: torch.device,
    train_loader,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    optimizer: Union[SGD, AdamW],
    amp_scaler: GradScaler,
    metrics_calculator: SemSegMetricsCalculator,
    world_size: int,
) -> tuple:
    """
    train a neural network model using a DALI dataloader with DDP

    ---------
    Arguments
    ---------
    model: DDP
        DDP-wrapped model object
    device: torch.device
        indicates torch device for this rank
    train_loader: DALIGenericIterator
        train data loader object (sharded for this rank)
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function
    optimizer: Union[SGD, AdamW]
        optimizer object to be used for training
    amp_scaler: GradScaler
        AMP scaler to be used for training
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator
    world_size: int
        the total number of processes

    -------
    Returns
    -------
    (train_loss, train_acc, train_f1, train_precision, train_recall, train_dice, train_miou, train_conf_mat):
        tuple of metrics for the train set
    """
    model.train()
    num_train_batches = len(train_loader)

    running_train_loss = torch.zeros(1, device=device)
    metrics_calculator.reset_metrics()

    for batch in train_loader:
        image = batch[0]["images"]
        label = batch[0]["labels"]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_logits = model(image)
            loss = criterion(pred_logits, label)

            pred_probs = softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_train_loss += loss

        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

    # reduce loss across all ranks
    train_loss = running_train_loss / num_train_batches
    train_loss = reduce_tensor(train_loss, world_size)

    (
        train_acc,
        train_f1,
        train_precision,
        train_recall,
        train_conf_mat,
        train_dice,
        train_miou,
    ) = metrics_calculator.compute_metrics()

    # reduce metrics across all ranks
    train_acc = reduce_tensor(train_acc, world_size)
    train_f1 = reduce_tensor(train_f1, world_size)
    train_precision = reduce_tensor(train_precision, world_size)
    train_recall = reduce_tensor(train_recall, world_size)
    train_dice = reduce_tensor(train_dice, world_size)
    train_miou = reduce_tensor(train_miou, world_size)

    # reduce confusion matrix across all ranks
    dist.all_reduce(train_conf_mat, op=dist.ReduceOp.SUM)
    # re-normalize the confusion matrix after summing
    row_sums = train_conf_mat.sum(dim=1, keepdim=True)
    row_sums = row_sums.clamp(min=1)
    train_conf_mat = train_conf_mat / row_sums

    train_loss = train_loss.clone().detach().cpu().numpy()
    train_acc = train_acc.clone().detach().cpu().numpy()
    train_f1 = train_f1.clone().detach().cpu().numpy()
    train_precision = train_precision.clone().detach().cpu().numpy()
    train_recall = train_recall.clone().detach().cpu().numpy()
    train_conf_mat = train_conf_mat.clone().detach().cpu().numpy()
    train_dice = train_dice.clone().detach().cpu().numpy()
    train_miou = train_miou.clone().detach().cpu().numpy()

    gc.collect()
    return (
        float(train_loss),
        float(train_acc),
        float(train_f1),
        float(train_precision),
        float(train_recall),
        float(train_dice),
        float(train_miou),
        train_conf_mat,
    )


def test_nn_ddp_dali(
    model: DDP,
    device: torch.device,
    test_loader,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    metrics_calculator: SemSegMetricsCalculator,
    world_size: int,
) -> tuple:
    """
    test a neural network model using a DALI dataloader with DDP

    ---------
    Arguments
    ---------
    model: DDP
        DDP-wrapped model object
    device: torch.device
        indicates torch device for this rank
    test_loader: DALIGenericIterator
        test data loader object (sharded for this rank)
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator
    world_size: int
        the total number of processes

    -------
    Returns
    -------
    (test_loss, test_acc, test_f1, test_precision, test_recall, test_dice, test_miou, test_conf_mat):
        tuple of metrics for the test set
    """
    model.eval()
    num_test_batches = len(test_loader)

    running_test_loss = torch.zeros(1, device=device)
    metrics_calculator.reset_metrics()

    with torch.no_grad():
        for batch in test_loader:
            image = batch[0]["images"]
            label = batch[0]["labels"]

            pred_logits = model(image)
            loss = criterion(pred_logits, label)

            pred_probs = softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_test_loss += loss

    # reduce loss across all ranks
    test_loss = running_test_loss / num_test_batches
    test_loss = reduce_tensor(test_loss, world_size)

    (
        test_acc,
        test_f1,
        test_precision,
        test_recall,
        test_conf_mat,
        test_dice,
        test_miou,
    ) = metrics_calculator.compute_metrics()

    # reduce metrics across all ranks
    test_acc = reduce_tensor(test_acc, world_size)
    test_f1 = reduce_tensor(test_f1, world_size)
    test_precision = reduce_tensor(test_precision, world_size)
    test_recall = reduce_tensor(test_recall, world_size)
    test_dice = reduce_tensor(test_dice, world_size)
    test_miou = reduce_tensor(test_miou, world_size)

    # reduce confusion matrix across all ranks
    dist.all_reduce(test_conf_mat, op=dist.ReduceOp.SUM)
    row_sums = test_conf_mat.sum(dim=1, keepdim=True)
    row_sums = row_sums.clamp(min=1)
    test_conf_mat = test_conf_mat / row_sums

    test_loss = test_loss.clone().detach().cpu().numpy()
    test_acc = test_acc.clone().detach().cpu().numpy()
    test_f1 = test_f1.clone().detach().cpu().numpy()
    test_precision = test_precision.clone().detach().cpu().numpy()
    test_recall = test_recall.clone().detach().cpu().numpy()
    test_conf_mat = test_conf_mat.clone().detach().cpu().numpy()
    test_dice = test_dice.clone().detach().cpu().numpy()
    test_miou = test_miou.clone().detach().cpu().numpy()

    gc.collect()
    return (
        float(test_loss),
        float(test_acc),
        float(test_f1),
        float(test_precision),
        float(test_recall),
        float(test_dice),
        float(test_miou),
        test_conf_mat,
    )


def train_sem_seg_ddp_pipeline(
    rank: int,
    world_size: int,
    dir_train_tfrecords: str,
    dir_test_tfrecords: str,
    dir_tmp_ckpt_model: str,
    batch_size: int,
    num_classes: int,
    labels_display_logs: List[str],
    class_weights: List[float],
    experiment_name: str,
    run_name: str,
    model_name: str,
    loss_fn: str,
    optimizer_name: str,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    num_epochs: int = 100,
    checkpoint_freq: int = 5,
    checkpoint_skip: int = 20,
    num_threads: int = 4,
    num_in_channels: int = 1,
    model_compile_mode: str = "normal",
    file_model_ckpt: Union[str, None] = None,
    model_task: str = "semantic_segmentation",
) -> None:
    """
    main model train pipeline for semantic segmentation task using DDP with DALI TFRecords

    ---------
    Arguments
    ---------
    rank: int
        the rank of the current process (also the GPU device id)
    world_size: int
        the total number of processes (number of GPUs)
    dir_train_tfrecords: str
        full path to the directory containing the train set tfrecord files
    dir_test_tfrecords: str
        full path to the directory containing the test set tfrecord files
    dir_tmp_ckpt_model: str
        full path to the directory where temporary model checkpoint file needs to be saved
    batch_size: int
        batch size to be used for training the model (per GPU)
    num_classes: int
        number of classes in the dataset for which the model needs to be trained
    labels_display_logs: List[str]
        the list of label names to be used for display in the confusion matrix figures
    class_weights: List[float]
        the list of class weights to be used in the loss function for handling class imbalance
    experiment_name: str
        MLFlow experiment name
    run_name: str
        MLFlow run name
    model_name: str
        model name
    loss_fn: str
        loss function to be used for training
    optimizer_name: str
        optimizer to be used for training
    learning_rate: float
        initial learning rate to be used for training (default: 1e-3)
    weight_decay: float
        weight decay to be used for training (default: 1e-5)
    num_epochs: int
        number of epochs for which the model needs to be trained (default: 100)
    checkpoint_freq: int
        model checkpoint frequency (default: 5)
    checkpoint_skip: int
        initial number of epochs for which the checkpoint need to be skipped (default: 20)
    num_threads: int
        number of workers to be used in the dataloader (default: 4)
    num_in_channels: int
        number of input channels (default: 1)
    model_compile_mode: str
        model compile mode (default: "normal")
    file_model_ckpt: Union[str, None]
        full path to intermediate model checkpoint that needs to be used in case of finetuning (default: None)
    model_task: str
        the task for which the model is being trained (default: "semantic_segmentation")
    """
    # initialize DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        logging.info(f"Model: {model_name}, model compile method: {model_compile_mode}")
        logging.info(f"DDP training with world_size={world_size}")

    path_dir_tmp_ckpt_model = Path(dir_tmp_ckpt_model)
    if rank == 0 and not path_dir_tmp_ckpt_model.is_dir():
        path_dir_tmp_ckpt_model.mkdir(parents=True)

    labels_display_logs = np.array(labels_display_logs)
    class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32)

    if rank == 0:
        logging.info(
            f"label names for display in experiment logs: {labels_display_logs}"
        )
        logging.info(f"class weights: {class_weights}")

    # create DALI dataloaders with sharding for DDP
    train_loader = build_dali_tfrecords_loader(
        dir_train_tfrecords,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=rank,
        is_train=True,
        shuffle=True,
        shard_id=rank,
        num_shards=world_size,
    )
    test_loader = build_dali_tfrecords_loader(
        dir_test_tfrecords,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=rank,
        is_train=False,
        shuffle=False,
        shard_id=rank,
        num_shards=world_size,
    )

    # build the model based on the selected option
    if model_name == "convnext_v2_tiny_deeplab_v3+":
        model = ConvNextV2TinyDeepLabV3Plus(num_in_channels, num_classes)
    elif model_name == "convnext_v2_base_deeplab_v3+":
        model = ConvNextV2BaseDeepLabV3Plus(num_in_channels, num_classes)
    elif model_name == "resnet34_unet":
        model = ResNet34UNet(num_in_channels, num_classes)
    elif model_name == "psa_resnet34_unet":
        model = PSAResNet34UNet(num_in_channels, num_classes)
    else:
        logging.error(f"unidentified option for model_name={model_name}")
        cleanup_ddp()
        return

    if model_compile_mode != "uncompiled":
        model = torch.compile(model, mode=model_compile_mode)

    if file_model_ckpt is not None:
        if rank == 0:
            logging.info(
                f"for model finetuning, loading the checkpoint from: {file_model_ckpt}"
            )
        checkpoint = torch.load(file_model_ckpt, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)

    model.to(device)

    # wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    if rank == 0:
        logging.info(f"device: {device}")

    metrics_calculator = SemSegMetricsCalculator(device, num_classes=num_classes)

    if loss_fn == "cross_entropy":
        criterion = CrossEntropyLoss(weight=class_weights.to(device))
    elif loss_fn == "focal":
        criterion = FocalLoss(weight=class_weights.to(device))
    else:
        logging.error("option not yet implemented")
        cleanup_ddp()
        return

    # create an optimizer object for the chosen optimizer
    if optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    # create an object for polynomial learning rate scheduler
    lr_scheduler = PolynomialLR(optimizer, total_iters=num_epochs, power=0.95)

    # create an object for AMP enabled training
    amp_scaler = GradScaler(device=device)

    # initialize mlflow experiment only on rank 0
    if rank == 0:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        logging.info(f"mlflow tracking uri: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        mlflow_run_tags = {
            "author": os.getenv("USER", "abhishek_r_s"),
            "experiment": experiment_name,
            "run": run_name,
            "distributed": "ddp",
            "world_size": str(world_size),
        }

    # start training
    if rank == 0:
        mlflow_context = mlflow.start_run(
            experiment_id=experiment.experiment_id, tags=mlflow_run_tags
        )
    else:
        mlflow_context = None

    try:
        if rank == 0:
            mlflow_context.__enter__()
            # log all the relevant params
            mlflow.log_param("optimization.optimizer_name", optimizer_name)
            mlflow.log_param("optimization.learning_rate", learning_rate)
            mlflow.log_param("optimization.weight_decay", weight_decay)
            mlflow.log_param("optimization.num_epochs", num_epochs)
            mlflow.log_param("optimization.loss_fn", loss_fn)
            mlflow.log_param("optimization.class_weights", class_weights)

            mlflow.log_param("dataset.num_in_channels", num_in_channels)
            mlflow.log_param("dataset.num_classes", num_classes)
            mlflow.log_param("dataset.dir_train_tfrecords", dir_train_tfrecords)
            mlflow.log_param("dataset.dir_test_tfrecords", dir_test_tfrecords)
            mlflow.log_param("dataset.batch_size_per_gpu", batch_size)
            mlflow.log_param("dataset.effective_batch_size", batch_size * world_size)
            mlflow.log_param("dataset.num_threads", num_threads)
            mlflow.log_param("dataset.class_label_mapping", labels_display_logs)

            mlflow.log_param("model.task", model_task)
            mlflow.log_param("model.model_compile_mode", model_compile_mode)
            mlflow.log_param("distributed.world_size", world_size)
            mlflow.log_param("distributed.backend", "nccl")

        for epoch in range(1, num_epochs + 1):
            time_start_epoch = time.time()

            (
                train_loss,
                train_acc,
                train_f1,
                train_precision,
                train_recall,
                train_dice,
                train_miou,
                train_conf_mat,
            ) = train_nn_ddp_dali(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                amp_scaler,
                metrics_calculator,
                world_size,
            )
            torch.cuda.empty_cache()

            (
                test_loss,
                test_acc,
                test_f1,
                test_precision,
                test_recall,
                test_dice,
                test_miou,
                test_conf_mat,
            ) = test_nn_ddp_dali(
                model,
                device,
                test_loader,
                criterion,
                metrics_calculator,
                world_size,
            )
            torch.cuda.empty_cache()

            time_end_epoch = time.time()
            lr_scheduler.step()

            # logging and checkpointing only on rank 0
            if rank == 0:
                logging.info(
                    f"epoch: {epoch} / {num_epochs}, time taken: {time_end_epoch - time_start_epoch:.2f} sec."
                )
                logging.info(
                    f"for train set, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, dice: {train_dice:.4f}, mIoU: {train_miou:.4f}"
                )
                logging.info(
                    f"for test set, loss: {test_loss:.4f}, accuracy: {test_acc:.4f}, f1: {test_f1:.4f}, precision: {test_precision:.4f}, recall: {test_recall:.4f}, dice: {test_dice:.4f}, mIoU: {test_miou:.4f}"
                )

                if (epoch > checkpoint_skip) and (epoch % checkpoint_freq == 0):
                    train_conf_matrix_fig = get_confusion_matrix_figure(
                        train_conf_mat,
                        labels_display_logs,
                    )
                    test_conf_matrix_fig = get_confusion_matrix_figure(
                        test_conf_mat,
                        labels_display_logs,
                    )

                    mlflow.log_figure(
                        train_conf_matrix_fig.figure_,
                        f"train_conf_matrix_{epoch}.png",
                    )
                    mlflow.log_figure(
                        test_conf_matrix_fig.figure_,
                        f"test_conf_matrix_{epoch}.png",
                    )

                    plt.close(train_conf_matrix_fig.figure_)
                    plt.close(test_conf_matrix_fig.figure_)

                    model_file_name = (
                        path_dir_tmp_ckpt_model / f"{model_name}_{epoch}.pth"
                    )
                    # save the underlying model state_dict (unwrap DDP)
                    torch.save(
                        {
                            "model_state_dict": model.module.state_dict(),
                            "model_class": model.module.__class__.__name__,
                            "model_config": {
                                "num_in_channels": num_in_channels,
                                "num_classes": num_classes,
                            },
                        },
                        model_file_name,
                    )
                    mlflow.log_artifact(
                        local_path=model_file_name, artifact_path="model_artifacts"
                    )

                    # delete the temp model file
                    model_file_name.unlink()

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("train_f1", train_f1, step=epoch)
                mlflow.log_metric("train_precision", train_precision, step=epoch)
                mlflow.log_metric("train_recall", train_recall, step=epoch)
                mlflow.log_metric("train_dice", train_dice, step=epoch)
                mlflow.log_metric("train_miou", train_miou, step=epoch)

                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("test_acc", test_acc, step=epoch)
                mlflow.log_metric("test_f1", test_f1, step=epoch)
                mlflow.log_metric("test_precision", test_precision, step=epoch)
                mlflow.log_metric("test_recall", test_recall, step=epoch)
                mlflow.log_metric("test_dice", test_dice, step=epoch)
                mlflow.log_metric("test_miou", test_miou, step=epoch)

            # synchronize all processes at the end of each epoch
            dist.barrier()
            gc.collect()

    finally:
        if rank == 0 and mlflow_context is not None:
            mlflow_context.__exit__(None, None, None)

    # remove the empty directory (only rank 0)
    if rank == 0:
        try:
            path_dir_tmp_ckpt_model.rmdir()
        except OSError:
            pass

    cleanup_ddp()
    return

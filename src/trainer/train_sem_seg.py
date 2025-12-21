import os
import gc
import time
import torch
import mlflow
import logging
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


from pathlib import Path
from torch import GradScaler
from torch.optim import SGD, AdamW
from typing import Union, Tuple, List
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR


from loss_func.focal_loss import FocalLoss
from data_handler.data_loader import get_dataloaders_for_training
from models.sem_seg_model import (
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
)
from metrics.compute_metrics import (
    SemSegMetricsCalculator,
    get_confusion_matrix_figure,
)


def train_nn(
    model: Union[ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus],
    device: torch.device,
    train_loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: Union[SGD, AdamW],
    amp_scaler: GradScaler,
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    train a neural network model
    """
    model.to(device)
    model.train()
    num_train_batches = len(train_loader)

    running_train_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    for msi_bands, label in train_loader:
        msi_bands = msi_bands.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_train_loss += loss

        # loss is scaled and then scaled gradients are created
        amp_scaler.scale(loss).backward()

        # apply the update
        amp_scaler.step(optimizer)

        # update the scaler for the next iteration
        amp_scaler.update()

    train_loss = float(running_train_loss) / num_train_batches
    (
        train_acc,
        train_f1,
        train_precision,
        train_recall,
        train_conf_mat,
        train_dice,
        train_miou,
    ) = metrics_calculator.compute_metrics()
    train_conf_mat = train_conf_mat.clone().detach().cpu().numpy()
    return (
        train_loss,
        float(train_acc),
        float(train_f1),
        float(train_precision),
        float(train_recall),
        float(train_dice),
        float(train_miou),
        train_conf_mat,
    )


def test_nn(
    model: Union[ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus],
    device: torch.device,
    test_loader: DataLoader,
    criterion: CrossEntropyLoss,
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    test a neural network model
    """
    model.to(device)
    model.eval()
    num_test_batches = len(test_loader)

    running_test_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    with torch.no_grad():
        for msi_bands, label in test_loader:
            msi_bands = msi_bands.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            pred_logits = model(msi_bands)
            loss = criterion(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_test_loss += loss

    test_loss = float(running_test_loss) / num_test_batches

    (
        test_acc,
        test_f1,
        test_precision,
        test_recall,
        test_conf_mat,
        test_dice,
        test_miou,
    ) = metrics_calculator.compute_metrics()
    test_conf_mat = test_conf_mat.clone().detach().cpu().numpy()
    return (
        test_loss,
        float(test_acc),
        float(test_f1),
        float(test_precision),
        float(test_recall),
        float(test_dice),
        float(test_miou),
        test_conf_mat,
    )


def train_sem_seg_pipeline(
    dir_train_images: str,
    dir_train_labels: str,
    dir_test_images: str,
    dir_test_labels: str,
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
    which_gpu: str = "0",
    num_workers: int = 8,
    num_in_channels: int = 1,
    model_compile_mode: str = "normal",
    file_model_ckpt: Union[str, None] = None,
    model_task: str = "semantic_segmentation",
) -> None:
    """
    main model train pipeline for vanilla style training for any type of NN
    """
    path_dir_tmp_ckpt_model = Path(dir_tmp_ckpt_model)
    if not path_dir_tmp_ckpt_model.is_dir():
        path_dir_tmp_ckpt_model.mkdir()

    path_dir_train_images = Path(dir_train_images)
    path_dir_test_images = Path(dir_test_images)

    list_train_images = [
        f.name for f in path_dir_train_images.glob("*png") if f.is_file()
    ]
    list_test_images = [
        f.name for f in path_dir_test_images.glob("*png") if f.is_file()
    ]

    logging.info(f"num train images: {len(list_train_images)}")
    logging.info(f"num test images: {len(list_test_images)}")

    labels_display_logs = np.array(labels_display_logs)
    logging.info(f"label names for display in experiment logs: {labels_display_logs}")

    class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32)
    logging.info(f"class weights: {class_weights}")

    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    # create train and test data loaders
    train_loader, test_loader = get_dataloaders_for_training(
        dir_train_images,
        dir_train_labels,
        dir_test_images,
        dir_test_labels,
        batch_size,
        num_workers=num_workers,
    )

    # build the model based on the selected option
    if model_name == "convnext_v2_tiny_deeplab_v3+":
        model = ConvNextV2TinyDeepLabV3Plus(
            num_in_channels,
            num_classes,
        )
    elif model_name == "convnext_v2_base_deeplab_v3+":
        model = ConvNextV2BaseDeepLabV3Plus(
            num_in_channels,
            num_classes,
        )
    else:
        logging.error(f"unidentified option for model_name={model_name}")

    # automatically choose the device
    if torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    device = torch.device(device_str)
    if model_compile_mode != "uncompiled":
        model = torch.compile(model, mode=model_compile_mode)

    if file_model_ckpt is not None:
        logging.info(
            f"for model finetuning, loading the checkpoint from: {file_model_ckpt}"
        )
        checkpoint = torch.load(file_model_ckpt, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)

    model.to(device)
    logging.info(device)

    metrics_calculator = SemSegMetricsCalculator(device, num_classes=num_classes)

    if loss_fn == "cross_entropy":
        criterion = CrossEntropyLoss(weight=class_weights.to(device))
    elif loss_fn == "focal":
        criterion = FocalLoss(weight=class_weights.to(device))
    else:
        logging.error("option not yet implemented")

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

    # initialize mlflow experiment and log running metrics for the experiment
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    logging.info(f"mlflow tracking uri: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    mlflow_run_tags = {
        "author": os.getenv("USER", "abhishek_r_s"),
        "experiment": experiment_name,
        "run": run_name,
    }

    # start training
    with mlflow.start_run(experiment_id=experiment.experiment_id, tags=mlflow_run_tags):
        # log all the relevant params
        mlflow.log_param("optimization.optimizer_name", optimizer_name)
        mlflow.log_param("optimization.learning_rate", learning_rate)
        mlflow.log_param("optimization.weight_decay", weight_decay)
        mlflow.log_param("optimization.num_epochs", num_epochs)
        mlflow.log_param("optimization.loss_fn", loss_fn)
        mlflow.log_param("optimization.class_weights", class_weights)

        mlflow.log_param("dataset.num_in_channels", num_in_channels)
        mlflow.log_param("dataset.num_classes", num_classes)
        mlflow.log_param("dataset.dir_train_images", dir_train_images)
        mlflow.log_param("dataset.dir_train_labels", dir_train_labels)
        mlflow.log_param("dataset.dir_test_images", dir_test_images)
        mlflow.log_param("dataset.dir_test_labels", dir_test_labels)
        mlflow.log_param("dataset.batch_size", batch_size)
        mlflow.log_param("dataset.num_workers", num_workers)
        mlflow.log_param("dataset.num_train_samples", len(list_train_images))
        mlflow.log_param("dataset.num_test_samples", len(list_test_images))
        mlflow.log_param("dataset.class_label_mapping", labels_display_logs)
        mlflow.log_text("\n".join(list_train_images), "dataset_list_train_images.txt")
        mlflow.log_text("\n".join(list_test_images), "dataset_list_test_images.txt")

        mlflow.log_param("model.task", model_task)
        mlflow.log_param("model.model_compile_mode", model_compile_mode)

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
            ) = train_nn(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                amp_scaler,
                metrics_calculator,
            )
            (
                test_loss,
                test_acc,
                test_f1,
                test_precision,
                test_recall,
                test_dice,
                test_miou,
                test_conf_mat,
            ) = test_nn(
                model,
                device,
                test_loader,
                criterion,
                metrics_calculator,
            )
            time_end_epoch = time.time()

            lr_scheduler.step()

            logging.info(
                f"epoch: {epoch} / {num_epochs}, time taken: {time_end_epoch - time_start_epoch:.2f} sec."
            )
            logging.info(
                f"for train set, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, dice: {train_dice:.4f}, mIoU: {train_miou:.4f}"
            )
            logging.info(
                f"for test set, loss: {test_loss:.4f}, accuracy: {test_acc:.4f}, f1: {test_f1:.4f}, precision: {test_precision:.4f}, recall: {test_recall:.4f}, dice: {test_dice:.4f}, mIoU: {test_miou:.4f}"
            )

            # keep track of the best model and its metrics
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
                    train_conf_matrix_fig.figure_, f"train_conf_matrix_{epoch}.png"
                )
                mlflow.log_figure(
                    test_conf_matrix_fig.figure_,
                    f"test_conf_matrix_{epoch}.png",
                )

                plt.close(train_conf_matrix_fig.figure_)
                plt.close(test_conf_matrix_fig.figure_)

                model_file_name = path_dir_tmp_ckpt_model / f"{model_name}_{epoch}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_class": model.__class__.__name__,
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

            gc.collect()
    # remove the empty directory
    path_dir_tmp_ckpt_model.rmdir()
    return

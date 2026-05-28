import os
import gc
import time
import torch
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
from torch import GradScaler
from torch.optim import SGD, AdamW
from typing import Union, List
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import PolynomialLR


from loss_func.focal_loss import FocalLoss
from data_handler.data_loader_dali import get_png_dataloaders
from trainer.train_common import train_nn_dali, test_nn_dali
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
    num_threads: int = 4,
    num_in_channels: int = 1,
    model_compile_mode: str = "normal",
    file_model_ckpt: Union[str, None] = None,
    model_task: str = "semantic_segmentation",
) -> None:
    """
    main model train pipeline for semantic segmentation task using DALI PNG dataloader

    ---------
    Arguments
    ---------
    dir_train_images: str
        full path to the directory containing the train set images
    dir_train_labels: str
        full path to the directory containing the train set labels
    dir_test_images: str
        full path to the directory containing the test set images
    dir_test_labels: str
        full path to the directory containing the test set labels
    dir_tmp_ckpt_model: str
        full path to the directory where temporary model checkpoint file needs to be saved
    batch_size: int
        batch size to be used for training the model
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
    which_gpu: str
        the GPU number to be used for training the model (default: "0")
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
    logging.info(f"Model: {model_name}, model compile method: {model_compile_mode}")
    path_dir_tmp_ckpt_model = Path(dir_tmp_ckpt_model)
    if not path_dir_tmp_ckpt_model.is_dir():
        path_dir_tmp_ckpt_model.mkdir()

    labels_display_logs = np.array(labels_display_logs)
    logging.info(f"label names for display in experiment logs: {labels_display_logs}")

    class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32)
    logging.info(f"class weights: {class_weights}")

    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    # create train and test data loaders using DALI PNG pipeline
    train_loader, test_loader = get_png_dataloaders(
        dir_train_images,
        dir_train_labels,
        dir_test_images,
        dir_test_labels,
        batch_size=batch_size,
        num_threads=num_threads,
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
    elif model_name == "resnet34_unet":
        model = ResNet34UNet(
            num_in_channels,
            num_classes,
        )
    elif model_name == "psa_resnet34_unet":
        model = PSAResNet34UNet(
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
        mlflow.log_param("dataset.num_threads", num_threads)
        mlflow.log_param("dataset.class_label_mapping", labels_display_logs)

        mlflow.log_param("model.task", model_task)
        mlflow.log_param("model.model_name", model_name)
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
            ) = train_nn_dali(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                amp_scaler,
                metrics_calculator,
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
            ) = test_nn_dali(
                model,
                device,
                test_loader,
                criterion,
                metrics_calculator,
            )
            torch.cuda.empty_cache()
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

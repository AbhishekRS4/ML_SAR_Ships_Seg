import gc
import torch
import numpy as np
import torch.nn.functional as F


from torch import GradScaler
from torch.optim import SGD, AdamW
from typing import Union, Tuple
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator


from loss_func.focal_loss import FocalLoss
from models.sem_seg_model import (
    ResNet34UNet,
    PSAResNet34UNet,
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
)
from metrics.compute_metrics import SemSegMetricsCalculator


def train_nn_dali(
    model: Union[
        ResNet34UNet, PSAResNet34UNet,
        ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus,
    ],
    device: torch.device,
    train_loader: DALIGenericIterator,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    optimizer: Union[SGD, AdamW],
    amp_scaler: GradScaler,
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
    """
    train a neural network model using a DALI dataloader

    ---------
    Arguments
    ---------
    model: Union[ResNet34UNet, PSAResNet34UNet, ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus]
        model object
    device: torch.device
        indicates torch device
    train_loader: DALIGenericIterator
        train data loader object
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function that needs to be used for training
    optimizer: Union[SGD, AdamW]
        optimizer object to be used for training
    amp_scaler: GradScaler
        AMP scaler to be used for training
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator

    -------
    Returns
    -------
    (train_loss, train_acc, train_f1, train_precision, train_recall, train_dice, train_miou, train_conf_mat):
        Tuple[float, float, float, float, float, float, float, np.ndarray]
        a tuple of metrics for the train set
    """
    model.to(device)
    model.train()
    num_train_batches = len(train_loader)

    running_train_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    for batch in train_loader:
        image = batch[0]["images"]
        label = batch[0]["labels"]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_logits = model(image)
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

    train_loss = running_train_loss.clone().detach().cpu().numpy() / num_train_batches
    (
        train_acc,
        train_f1,
        train_precision,
        train_recall,
        train_conf_mat,
        train_dice,
        train_miou,
    ) = metrics_calculator.compute_metrics()

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


def test_nn_dali(
    model: Union[
        ResNet34UNet, PSAResNet34UNet,
        ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus,
    ],
    device: torch.device,
    test_loader: DALIGenericIterator,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
    """
    test a neural network model using a DALI dataloader

    ---------
    Arguments
    ---------
    model: Union[ResNet34UNet, PSAResNet34UNet, ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus]
        model object
    device: torch.device
        indicates torch device
    test_loader: DALIGenericIterator
        test data loader object
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function that needs to be used to compute loss for the test set
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator

    -------
    Returns
    -------
    (test_loss, test_acc, test_f1, test_precision, test_recall, test_dice, test_miou, test_conf_mat):
        Tuple[float, float, float, float, float, float, float, np.ndarray]
        a tuple of metrics for the test set
    """
    model.to(device)
    model.eval()
    num_test_batches = len(test_loader)

    running_test_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    with torch.no_grad():
        for batch in test_loader:
            image = batch[0]["images"]
            label = batch[0]["labels"]

            pred_logits = model(image)
            loss = criterion(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_test_loss += loss

    test_loss = running_test_loss.clone().detach().cpu().numpy() / num_test_batches

    (
        test_acc,
        test_f1,
        test_precision,
        test_recall,
        test_conf_mat,
        test_dice,
        test_miou,
    ) = metrics_calculator.compute_metrics()

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


def train_nn(
    model: Union[
        ResNet34UNet, PSAResNet34UNet,
        ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus,
    ],
    device: torch.device,
    train_loader: DataLoader,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    optimizer: Union[SGD, AdamW],
    amp_scaler: GradScaler,
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
    """
    train a neural network model using a PyTorch DataLoader

    ---------
    Arguments
    ---------
    model: Union[ResNet34UNet, PSAResNet34UNet, ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus]
        model object
    device: torch.device
        indicates torch device
    train_loader: DataLoader
        train data loader object
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function that needs to be used for training
    optimizer: Union[SGD, AdamW]
        optimizer object to be used for training
    amp_scaler: GradScaler
        AMP scaler to be used for training
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator

    -------
    Returns
    -------
    (train_loss, train_acc, train_f1, train_precision, train_recall, train_dice, train_miou, train_conf_mat):
        Tuple[float, float, float, float, float, float, float, np.ndarray]
        a tuple of metrics for the train set
    """
    model.to(device)
    model.train()
    num_train_batches = len(train_loader)

    running_train_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    for image, label in train_loader:
        image = image.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_logits = model(image)
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

    train_loss = running_train_loss.clone().detach().cpu().numpy() / num_train_batches
    (
        train_acc,
        train_f1,
        train_precision,
        train_recall,
        train_conf_mat,
        train_dice,
        train_miou,
    ) = metrics_calculator.compute_metrics()

    train_acc = train_acc.clone().detach().cpu().numpy()
    train_f1 = train_f1.clone().detach().cpu().numpy()
    train_precision = train_precision.clone().detach().cpu().numpy()
    train_recall = train_recall.clone().detach().cpu().numpy()
    train_conf_mat = train_conf_mat.clone().detach().cpu().numpy()
    train_dice = train_dice.clone().detach().cpu().numpy()
    train_miou = train_miou.clone().detach().cpu().numpy()
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


def test_nn(
    model: Union[
        ResNet34UNet, PSAResNet34UNet,
        ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus,
    ],
    device: torch.device,
    test_loader: DataLoader,
    criterion: Union[CrossEntropyLoss, FocalLoss],
    metrics_calculator: SemSegMetricsCalculator,
) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
    """
    test a neural network model using a PyTorch DataLoader

    ---------
    Arguments
    ---------
    model: Union[ResNet34UNet, PSAResNet34UNet, ConvNextV2TinyDeepLabV3Plus, ConvNextV2BaseDeepLabV3Plus]
        model object
    device: torch.device
        indicates torch device
    test_loader: DataLoader
        test data loader object
    criterion: Union[CrossEntropyLoss, FocalLoss]
        criterion object for the loss function that needs to be used to compute loss for the test set
    metrics_calculator: SemSegMetricsCalculator
        object for the metrics calculator

    -------
    Returns
    -------
    (test_loss, test_acc, test_f1, test_precision, test_recall, test_dice, test_miou, test_conf_mat):
        Tuple[float, float, float, float, float, float, float, np.ndarray]
        a tuple of metrics for the test set
    """
    model.to(device)
    model.eval()
    num_test_batches = len(test_loader)

    running_test_loss = torch.zeros(1).to(device)
    metrics_calculator.reset_metrics()

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            pred_logits = model(image)
            loss = criterion(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(label, pred_label)

            running_test_loss += loss

    test_loss = running_test_loss.clone().detach().cpu().numpy() / num_test_batches

    (
        test_acc,
        test_f1,
        test_precision,
        test_recall,
        test_conf_mat,
        test_dice,
        test_miou,
    ) = metrics_calculator.compute_metrics()

    test_acc = test_acc.clone().detach().cpu().numpy()
    test_f1 = test_f1.clone().detach().cpu().numpy()
    test_precision = test_precision.clone().detach().cpu().numpy()
    test_recall = test_recall.clone().detach().cpu().numpy()
    test_conf_mat = test_conf_mat.clone().detach().cpu().numpy()
    test_dice = test_dice.clone().detach().cpu().numpy()
    test_miou = test_miou.clone().detach().cpu().numpy()
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

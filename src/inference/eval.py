import os
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F


from typing import List
from pathlib import Path, PosixPath
from skimage.io import imsave, imread


from inference.timer import gpu_timer
from data_handler.data_processing import preprocess_image_tensor
from models.sem_seg_model import (
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
    ResNet34UNet,
    PSAResNet34UNet,
)
from metrics.compute_metrics import SemSegMetricsCalculator


def eval_pipeline(
    dir_test_images: str,
    dir_test_labels: str,
    file_model_ckpt: str,
    model_name: str,
    model_compile_mode: str,
    which_gpu: str = "0",
    num_classes: int = 2,
) -> None:
    """
    function for inference pipeline with normal torch model ckpt

    ---------
    Arguments
    ---------
    dir_test_images: str
        full path to the directory containing the test images
    dir_predictions: str
        full path to the directory where the predicted labels need to be saved
    file_model_ckpt: str
        full path to the model checkpoint
    model_name: str
        model name
    model_compile_mode: str
        model compile mode
    which_gpu: str
        GPU number on which the model inference needs to be run (default: "0")
    num_classes: int
        number of classes for computing the metrics (default: 2)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    path_file_model_ckpt = Path(file_model_ckpt)
    path_dir_test_images = Path(dir_test_images)
    path_dir_test_labels = Path(dir_test_labels)

    # automatically choose the device
    if torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    device = torch.device(device_str)

    model_checkpoint = torch.load(path_file_model_ckpt)
    model_state_dict = model_checkpoint["model_state_dict"]

    if model_name == "convnext_v2_tiny_deeplab_v3+":
        model = ConvNextV2TinyDeepLabV3Plus(**model_checkpoint["model_config"])
    elif model_name == "convnext_v2_base_deeplab_v3+":
        model = ConvNextV2BaseDeepLabV3Plus(**model_checkpoint["model_config"])
    elif model_name == "resnet34_unet":
        model = ResNet34UNet(**model_checkpoint["model_config"])
    elif model_name == "psa_resnet34_unet":
        model = PSAResNet34UNet(**model_checkpoint["model_config"])
    else:
        logging.error(f"unidentified option for model_name={model_name}")

    if model_compile_mode != "uncompiled":
        if model_compile_mode != "normal":
            model = torch.compile(model, mode=model_compile_mode)
        else:
            model = torch.compile(model)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    list_test_images = sorted(
        [f for f in path_dir_test_images.glob("*png") if f.is_file()]
    )

    metrics_calculator = SemSegMetricsCalculator(device, num_classes=num_classes)
    metrics_calculator.reset_metrics()

    for file_test_img in list_test_images:
        file_name_pred = file_test_img.name
        test_img_arr = imread(file_test_img)

        test_lbl_arr = imread(path_dir_test_labels / file_test_img.name)
        test_lbl_tensor = torch.from_numpy(test_lbl_arr)
        test_lbl_tensor = test_lbl_tensor.to(device, dtype=torch.long)
        test_lbl_tensor = torch.unsqueeze(test_lbl_tensor, dim=0)

        test_img_tensor = torch.from_numpy(test_img_arr[:, :, 0])
        test_img_tensor = test_img_tensor.to(device, dtype=torch.float32)
        test_img_tensor = torch.unsqueeze(
            torch.unsqueeze(test_img_tensor, dim=0), dim=0
        )
        test_img_tensor = preprocess_image_tensor(test_img_tensor)

        with torch.no_grad():
            pred_logits, time_taken = gpu_timer(lambda: model(test_img_tensor))
            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            metrics_calculator.update_metrics(test_lbl_tensor, pred_label)

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

    logging.info(f"Path to images: {path_dir_test_images}")
    logging.info(f"Path to labels: {path_dir_test_labels}")
    logging.info("Metrics")

    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info(f"F1-score: {test_f1:.4f}")
    logging.info(f"Precision: {test_precision:.4f}")
    logging.info(f"Recall: {test_recall:.4f}")
    logging.info(f"Dice: {test_dice:.4f}")
    logging.info(f"mIOU: {test_miou:.4f}")
    logging.info(f"Confusion Matrix:")
    logging.info(test_conf_mat)
    return

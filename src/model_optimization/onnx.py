import os
import time
import torch
import logging


from pathlib import Path
from models.sem_seg_model import (
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
    ResNet34UNet,
    PSAResNet34UNet,
)


def optimize_model_with_onnx(
    file_model_ckpt: str,
    model_name: str,
    image_height: int,
    image_width: int,
    model_compile_mode: str = "reduce-overhead",
) -> None:
    """
    Docstring for optimize_model_with_onnx

    ---------
    Arguments
    ---------
    file_model_ckpt: str
        full path to the model checkpoint file
    model_name: str
        model name for which the achitecture needs to be initialized
    image_height: int
        image height
    image_width: int
        image width
    model_compile_mode: str
        model compile mode (default: "reduce-overhead")
    """
    path_file_model_ckpt = Path(file_model_ckpt)
    if not path_file_model_ckpt.is_file():
        logging.error(f"file not found: {path_file_model_ckpt}")

    logging.info(f"loading model ckpt file from: {path_file_model_ckpt}")
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
        logging.error(f"unknown option for (model_name={model_name})")

    if model_compile_mode != "uncompiled":
        model = torch.compile(model, mode=model_compile_mode)

    model.load_state_dict(model_state_dict)
    model.eval()

    original_model = model._orig_mod
    original_model.load_state_dict(model._orig_mod.state_dict())
    original_model.eval()

    num_in_channels = model_checkpoint["model_config"]["num_in_channels"]

    example_inputs = (
        torch.randn(1, num_in_channels, image_height, image_width, dtype=torch.float32),
    )

    path_file_model_ckpt_onnx = (
        path_file_model_ckpt.parent
        / path_file_model_ckpt.name.replace(".pth", ".onnx").replace(".pt", ".onnx")
    )

    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            original_model,  # model to export
            example_inputs,  # inputs of the model
            export_params=True,  # include model params as well
            opset_version=17,  # ONNX version
            do_constant_folding=True,  # Optimize using constant folding
            input_names=["input"],  # Rename inputs for the ONNX model
            output_names=["output"],  # Rename outpus for the ONNX model
            dynamo=True,  # True or False to select the exporter to use
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        onnx_program.optimize()
        onnx_program.save(path_file_model_ckpt_onnx)
    logging.info(
        f"optimized model with AOT Inductor is saved to: {path_file_model_ckpt_onnx}"
    )
    return

import os
import time
import torch
import logging
import torch._dynamo
import torch_tensorrt

from pathlib import Path
from models.sem_seg_model import (
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
    ResNet34UNet,
    PSAResNet34UNet,
)


def optimize_model_with_tensorrt(
    file_model_ckpt: str,
    model_name: str,
    image_height: int,
    image_width: int,
    precision: str = "fp32",
    which_gpu: str = "0",
    model_compile_mode: str = "reduce-overhead",
) -> None:
    """
    function for optimizing the model with TensorRT for inference and saving the exported program

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
    which_gpu: str
        indicates the GPU number that needs to be used (default: "0")
    model_compile_mode: str
        model compile mode (default: "reduce-overhead")
    """
    torch._dynamo.config.suppress_errors = True

    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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
    model = model.to(device=device)

    # path to output exported program
    path_file_trt_model = (
        path_file_model_ckpt.parent
        / f"{precision}_trt_optimized_{path_file_model_ckpt.name.split('.')[0]}.ep"
    )

    num_in_channels = model_checkpoint["model_config"]["num_in_channels"]

    if precision == "fp32":
        example_inputs = [
            torch.randn(
                (1, num_in_channels, image_height, image_width), dtype=torch.float32
            ).cuda(),
        ]
    elif precision == "mixed":
        """
        model = model.half()
        example_inputs = [
            torch.randn(
                (1, num_in_channels, image_height, image_width), dtype=torch.half
            ).cuda(),
        ]
        """

        example_inputs = [
            torch.randn(
                (1, num_in_channels, image_height, image_width), dtype=torch.float32
            ).cuda(),
        ]
    else:
        logging.error(f"Unidentified option for precision: {precision}")

    exp_program = torch.export.export(model, tuple(example_inputs))

    if precision == "fp32":
        trt_model = torch_tensorrt.dynamo.compile(
            exp_program,
            example_inputs,
            optimization_level=5,
        )
    elif precision == "mixed":
        trt_model = torch_tensorrt.dynamo.compile(
            exp_program,
            example_inputs,
            enabled_precisions={torch.float32, torch.half},
            optimization_level=5,
        )
    else:
        logging.error(f"Unidentified option for precision: {precision}")

    torch_tensorrt.save(trt_model, path_file_trt_model, inputs=example_inputs)

    logging.info(f"optimized model with TensorRT is saved to: {path_file_trt_model}")

    return

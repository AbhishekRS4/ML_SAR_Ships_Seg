import os
import time
import torch
import logging

from pathlib import Path
from models.sem_seg_model import (
    ConvNextV2TinyDeepLabV3Plus,
    ConvNextV2BaseDeepLabV3Plus,
)


def optimize_model_with_aot_inductor(
    file_model_ckpt: str,
    model_name: str,
    image_height: int,
    image_width: int,
    which_gpu: str = "0",
    model_compile_mode: str = "reduce-overhead",
) -> None:
    """
    function for optimizing the model with AOT inductor for inference and saving the model file
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
    else:
        logging.error(f"unknown option for (model_name={model_name})")

    if model_compile_mode != "uncompiled":
        if model_compile_mode != "normal":
            model = torch.compile(model, mode=model_compile_mode)
        else:
            model = torch.compile(model)
    model.load_state_dict(model_state_dict)
    model.eval()

    path_file_aot_inductor_model = (
        path_file_model_ckpt.parent
        / f"aot_optimized_{path_file_model_ckpt.name.split('.')[0]}.pt2"
    )
    # path_file_aot_inductor_model = os.path.join(str(path_file_model_ckpt.parent), f"aot_optimized_{path_file_model_ckpt.name.split('.')[0]}.pt2")

    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    with torch.inference_mode():
        inductor_configs = {}

        if torch.cuda.is_available():
            device = "cuda"
            inductor_configs["max_autotune"] = True
        else:
            device = "cpu"

        model = model.to(device=device)
        example_inputs = (torch.randn(1, 1, image_height, image_width, device=device),)

        exported_program = torch.export.export(
            model,
            example_inputs,
        )

        _ = torch._inductor.aoti_compile_and_package(
            exported_program,
            package_path=str(path_file_aot_inductor_model),
            inductor_configs=inductor_configs,
        )
        # do not use pathlib PosixPaths, the AOT inductor optimized model saving will fail as it expects a string buffer or None
        # also, the file extension needs to be .pt2
    logging.info(
        f"optimized model with AOT Inductor is saved to: {path_file_aot_inductor_model}"
    )
    return

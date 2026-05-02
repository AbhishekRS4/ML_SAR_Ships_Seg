import os
import time
import torch
import logging
import numpy as np

import onnxruntime as ort
import torch.nn.functional as F


from typing import List
from pathlib import Path, PosixPath
from skimage.io import imsave, imread


from inference.timer import gpu_timer
from data_handler.data_processing import preprocess_image_tensor


def onnx_inference_pipeline(
    dir_test_images: str,
    dir_predictions: str,
    file_model_onnx: str,
    which_gpu: str = "0",
) -> None:
    """
    function for inference with ONNX model ckpt

    ---------
    Arguments
    ---------
    dir_test_images: str
        full path to the directory containing the test images
    dir_predictions: str
        full path to the directory where the predicted labels need to be saved
    file_model_onnx: str,
        full path to the ONNX model runtime
    which_gpu: str
        GPU number on which the model inference needs to be run (default: "0")
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    path_dir_test_images = Path(dir_test_images)
    path_dir_predictions = Path(dir_predictions)

    if not path_dir_predictions.is_dir():
        path_dir_predictions.mkdir()

    list_test_images = sorted(
        [f for f in path_dir_test_images.glob("*png") if f.is_file()]
    )

    logging.info(f"Loading ONNX runtime session from: {file_model_onnx}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",  # Fallback to CPU if CUDA fails
    ]
    ort_sess = ort.InferenceSession(
        file_model_onnx, session_options, providers=providers
    )

    ort_input_name = ort_sess.get_inputs()[0].name

    for file_test_img in list_test_images:
        file_name_pred = file_test_img.name
        test_img_arr = imread(file_test_img)
        test_img_arr = np.expand_dims(test_img_arr[:, :, 0], axis=(0, 1)).astype(
            np.float32
        )
        test_img_arr = test_img_arr / 255.0

        outputs, time_taken = gpu_timer(
            lambda: ort_sess.run(None, {ort_input_name: test_img_arr})
        )

        pred_label = np.argmax(outputs[0], axis=1)
        pred_label = np.squeeze(pred_label).astype(np.uint8)
        imsave(path_dir_predictions / file_name_pred, pred_label)
        logging.info(
            f"Time taken for inference with the optimized ONNX runtime model is {time_taken:.2f} ms"
        )
    return


def onnx_inference_pipeline_torch_io_binding(
    dir_test_images: str,
    dir_predictions: str,
    file_model_onnx: str,
    num_in_channels: int = 1,
    image_height: int = 800,
    image_width: int = 800,
    num_classes: int = 2,
    which_gpu: str = "0",
) -> None:
    """
    function for inference with ONNX model ckpt with torch tensor I/O binding

    ---------
    Arguments
    ---------
    dir_test_images: str
        full path to the directory containing the test images
    dir_predictions: str
        full path to the directory where the predicted labels need to be saved
    file_model_onnx: str,
        full path to the ONNX model runtime
    num_in_channels: int
        num input channels (default: 1)
    image_height: int
        image height for creating tensor buffers (default: 800)
    image_width: int
        image width for creating tensor buffers (default: 800)
    num_classes: int
        num classes for creating tensor buffers (default: 2)
    which_gpu: str
        GPU number on which the model inference needs to be run (default: "0")
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    path_dir_test_images = Path(dir_test_images)
    path_dir_predictions = Path(dir_predictions)

    if not path_dir_predictions.is_dir():
        path_dir_predictions.mkdir()

    list_test_images = sorted(
        [f for f in path_dir_test_images.glob("*png") if f.is_file()]
    )
    # list_test_images = list_test_images[0:10]

    logging.info(f"Loading ONNX runtime session from: {file_model_onnx}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",  # Fallback to CPU if CUDA fails
    ]

    ort_sess = ort.InferenceSession(
        file_model_onnx, session_options, providers=providers
    )
    mem_info = ort_sess.get_input_memory_infos()[0]

    X_tensor = torch.randn(
        1,
        num_in_channels,
        image_height,
        image_width,
        device="cuda",
        dtype=torch.float32,
    )
    Y_tensor = torch.empty(
        1, num_classes, image_height, image_width, device="cuda", dtype=torch.float32
    )

    ort_input_name = ort_sess.get_inputs()[0].name
    ort_output_name = ort_sess.get_outputs()[0].name

    io_binding = ort_sess.io_binding()
    io_binding.bind_input(
        name=ort_input_name,
        device_type="cuda",
        device_id=mem_info.device_id,
        element_type=np.float32,
        shape=tuple(X_tensor.shape),
        buffer_ptr=X_tensor.data_ptr(),
    )
    io_binding.bind_output(
        name=ort_output_name,
        device_type="cuda",
        device_id=mem_info.device_id,
        element_type=np.float32,
        shape=tuple(Y_tensor.shape),
        buffer_ptr=Y_tensor.data_ptr(),
    )

    for file_test_img in list_test_images:
        file_name_pred = file_test_img.name
        test_img_arr = imread(file_test_img)
        test_img_arr = np.expand_dims(test_img_arr[:, :, 0], axis=(0, 1)).astype(
            np.float32
        )
        test_img_arr = test_img_arr / 255.0

        test_img_tensor = torch.from_numpy(test_img_arr)
        X_tensor.copy_(test_img_tensor.to("cuda").contiguous())

        _, time_taken = gpu_timer(lambda: ort_sess.run_with_iobinding(io_binding))

        outputs = io_binding.copy_outputs_to_cpu()
        pred_label = np.argmax(outputs[0], axis=1)
        pred_label = np.squeeze(pred_label).astype(np.uint8)
        imsave(path_dir_predictions / file_name_pred, pred_label)
        logging.info(
            f"Time taken for inference with the optimized ONNX runtime model with IO binding is {time_taken:.2f} ms"
        )

    return

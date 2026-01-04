import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from pathlib import Path, PosixPath
from concurrent.futures import ProcessPoolExecutor


def generate_n_save_vis(
    path_file_img: PosixPath,
    path_file_gt_lbl: PosixPath,
    path_file_pred_lbl: PosixPath,
    path_file_vis: PosixPath,
) -> None:
    """
    Docstring for generate_n_save_vis

    path_file_img: PosixPath
        full path to the image file
    path_file_gt_lbl: PosixPath
        full path to the gt label file
    path_file_pred_lbl: PosixPath
        full path to the pred label file
    path_file_vis: PosixPath
        full path to the visualization file that needs to be saved
    """
    file_name = path_file_img.name

    img_arr = imread(path_file_img)
    gt_lbl_arr = imread(path_file_gt_lbl)
    pred_lbl_arr = imread(path_file_pred_lbl)

    fig, ax = plt.subplot_mosaic("ABC", figsize=(16, 6))
    fig.suptitle(file_name, fontsize=16)

    ax["A"].imshow(img_arr)
    ax["A"].set_title("Image", fontsize=16)
    ax["B"].imshow(gt_lbl_arr, cmap="gray")
    ax["B"].set_title("GT label", fontsize=16)
    ax["C"].imshow(pred_lbl_arr, cmap="gray")
    ax["C"].set_title("Pred label", fontsize=16)

    plt.savefig(path_file_vis)
    plt.close()
    return


def save_visualization_pipeline(
    dir_images: str,
    dir_gt_lbls: str,
    dir_pred_lbls: str,
    dir_vis: str,
    num_workers: int = 4,
) -> None:
    """
    Docstring for save_visualization_pipeline

    dir_images: str
        full path to the directory containing the images
    dir_gt_lbl: str
        full path to the directory containing the gt labels
    dir_pred_lbl: str
        full path to the directory containing the pred labels
    dir_vis: str
        full path to the directory where the visualizations need to be saved
    num_workers: int
        num workers to be used in ProcessPoolExecutor (default: 4)
    """
    path_dir_images = Path(dir_images)
    path_dir_gt_lbls = Path(dir_gt_lbls)
    path_dir_pred_lbls = Path(dir_pred_lbls)
    path_dir_vis = Path(dir_vis)

    if not path_dir_vis.is_dir():
        path_dir_vis.mkdir()

    list_images = [f.name for f in path_dir_images.glob("*png") if f.is_file()]

    start_time = time.time()
    with ProcessPoolExecutor(
        max_workers=min(num_workers, len(list_images))
    ) as pp_executor:
        for file_img in list_images:
            path_file_img = path_dir_images / file_img
            path_file_gt_lbl = path_dir_gt_lbls / file_img
            path_file_pred_lbl = path_dir_pred_lbls / file_img
            path_file_vis = path_dir_vis / file_img

            pp_executor.submit(
                generate_n_save_vis,
                path_file_img,
                path_file_gt_lbl,
                path_file_pred_lbl,
                path_file_vis,
            )
    end_time = time.time()

    logging.info(
        f"Total time taken to save {len(list_images)} visualizations: {end_time - start_time} sec."
    )
    logging.info(f"Visualizations saved to {path_dir_vis}")
    return

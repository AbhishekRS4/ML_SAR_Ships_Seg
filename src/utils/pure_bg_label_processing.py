import time
import logging
import numpy as np

from PIL import Image
from pathlib import Path, PosixPath
from concurrent.futures import ThreadPoolExecutor


def save_pure_bg_label_per_image(
    path_file_pure_bg_image: PosixPath,
    path_dir_pure_bg_out_labels: PosixPath,
    img_height: int,
    img_width: int,
) -> None:
    """
    save label for pure background image for every image
    """
    file_name = path_file_pure_bg_image.name

    lbl_image_arr = np.zeros((img_height, img_width), dtype=np.uint8)
    lbl_image = Image.fromarray(lbl_image_arr, mode="L")
    lbl_image.save(path_dir_pure_bg_out_labels / file_name)
    return


def create_pure_bg_lbl_images(
    dir_pure_bg_images: str,
    dir_pure_bg_out_labels: str,
    image_height: int = 800,
    image_width: int = 800,
    num_workers: int = 4,
) -> None:
    """
    create pure background image labels
    """
    path_dir_pure_bg_images = Path(dir_pure_bg_images)
    list_pure_bg_images = [
        f for f in path_dir_pure_bg_images.glob("*png") if f.is_file()
    ]

    path_dir_pure_bg_out_labels = Path(dir_pure_bg_out_labels)
    if not path_dir_pure_bg_out_labels.is_dir():
        path_dir_pure_bg_out_labels.mkdir()

    logging.info(
        f"num pure bg images for which labels need to be created: {len(list_pure_bg_images)}"
    )

    time_start = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as tp_executor:
        for file_pure_bg_image in list_pure_bg_images:
            tp_executor.submit(
                save_pure_bg_label_per_image,
                file_pure_bg_image,
                path_dir_pure_bg_out_labels,
                image_height,
                image_width,
            )
    time_end = time.time()
    logging.info(f"saved pure bg labels in: {path_dir_pure_bg_out_labels}")
    logging.info(f"total time taken: {time_end - time_start} sec.")

    return

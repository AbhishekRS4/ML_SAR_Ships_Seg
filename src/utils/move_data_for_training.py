import shutil
import logging

from copy import deepcopy
from pathlib import Path, PosixPath
from sklearn.model_selection import train_test_split


def move_bg_images_n_labels(
    dir_pure_bg_images: str,
    dir_pure_bg_labels: str,
    dir_train_images: str,
    dir_train_labels: str,
    dir_test_images: str,
    dir_test_labels: str,
    test_size: float = 0.35,
) -> None:
    """
    split the background images into train and test sets and
    move them to the appropriate directories
    """
    path_dir_pure_bg_images = Path(dir_pure_bg_images)
    path_dir_pure_bg_labels = Path(dir_pure_bg_labels)

    path_dir_train_images = Path(dir_train_images)
    path_dir_train_labels = Path(dir_train_labels)

    path_dir_test_images = Path(dir_test_images)
    path_dir_test_labels = Path(dir_test_labels)

    list_images = sorted(
        [f.name for f in path_dir_pure_bg_images.glob("*png") if f.is_file()]
    )

    list_train_images, list_test_images = train_test_split(
        list_images,
        test_size=test_size,
    )

    for file_train_image in list_train_images:
        path_train_image_src = path_dir_pure_bg_images / file_train_image
        path_train_image_target = path_dir_train_images / file_train_image
        shutil.move(path_train_image_src, path_train_image_target)

        path_train_label_src = path_dir_pure_bg_labels / file_train_image
        path_train_label_target = path_dir_train_labels / file_train_image
        shutil.move(path_train_label_src, path_train_label_target)

    logging.info(
        f"Moved {len(list_train_images)} pure background train images to {path_dir_train_images}"
    )
    logging.info(
        f"Moved {len(list_train_images)} pure background train labels to {path_dir_train_labels}"
    )

    for file_test_image in list_test_images:
        path_test_image_src = path_dir_pure_bg_images / file_test_image
        path_test_image_target = path_dir_test_images / file_test_image
        shutil.move(path_test_image_src, path_test_image_target)

        path_test_label_src = path_dir_pure_bg_labels / file_test_image
        path_test_label_target = path_dir_test_labels / file_test_image
        shutil.move(path_test_label_src, path_test_label_target)

    logging.info(
        f"Moved {len(list_test_images)} pure background test images to {path_dir_test_images}"
    )
    logging.info(
        f"Moved {len(list_test_images)} pure background test labels to {path_dir_test_labels}"
    )
    return


def move_ship_images(
    dir_ship_images: str,
    dir_train_images: str,
    dir_train_labels: str,
    dir_test_images: str,
    dir_test_labels: str,
) -> None:
    """
    move the ship images to its respective train and test directories
    """
    path_dir_ship_images = Path(dir_ship_images)

    path_dir_train_images = Path(dir_train_images)
    path_dir_train_labels = Path(dir_train_labels)

    path_dir_test_images = Path(dir_test_images)
    path_dir_test_labels = Path(dir_test_labels)

    list_train_labels = [
        f.name for f in path_dir_train_labels.glob("*png") if f.is_file()
    ]
    list_test_labels = [
        f.name for f in path_dir_test_labels.glob("*png") if f.is_file()
    ]

    for file_train_label in list_train_labels:
        path_train_image_src = path_dir_ship_images / file_train_label
        path_train_image_target = path_dir_train_images / file_train_label
        shutil.move(path_train_image_src, path_train_image_target)

    logging.info(
        f"Moved {len(list_train_labels)} train images with ships to {path_dir_train_images}"
    )

    for file_test_label in list_test_labels:
        path_test_image_src = path_dir_ship_images / file_test_label
        path_test_image_target = path_dir_test_images / file_test_label
        shutil.move(path_test_image_src, path_test_image_target)

    logging.info(
        f"Moved {len(list_test_labels)} test images with ships to {path_dir_test_images}"
    )
    return

import logging
import numpy as np

from pathlib import Path
from skimage.io import imread


def compute_class_weights(
    dir_labels: str,
) -> None:
    """
    compute the class weights in the training set
    """
    path_dir_labels = Path(dir_labels)
    label_files_generator = path_dir_labels.glob("*.png")

    logging.info(f"dir path train labels: {path_dir_labels}")

    dict_labels_sum = {}

    for path_file_label in label_files_generator:
        if path_file_label.is_file():
            labels_arr = imread(path_file_label)
            unique_classes, count_classes = np.unique(labels_arr, return_counts=True)

            for index in range(len(unique_classes)):
                current_count = dict_labels_sum.get(unique_classes[index], 0)
                total_count = current_count + count_classes[index]
                dict_labels_sum[unique_classes[index]] = total_count

    logging.info("class imbalance information")
    unique_classes, count_classes = [], []

    dict_labels_sum = dict(sorted(dict_labels_sum.items()))

    unique_classes = list(dict_labels_sum.keys())
    count_classes = list(dict_labels_sum.values())

    unique_classes = np.array(unique_classes)
    count_classes = np.array(count_classes)

    logging.info("Unique classes")
    logging.info(unique_classes)

    logging.info("Counts per class")
    logging.info(count_classes)

    total_samples = np.sum(count_classes)
    class_weights = total_samples / (count_classes.shape[0] * count_classes)
    logging.info(f"balanced class weights: {class_weights}")
    return

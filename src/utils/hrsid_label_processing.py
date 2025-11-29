import os
import json
import time
import logging
import numpy as np

from pathlib import Path, PosixPath
from typing import Union, List, Dict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from pycocotools import mask as mask_util


def save_label_per_image(
    annotations: List[Dict],
    file_name: str,
    height: int,
    width: int,
    path_output_lbls: PosixPath,
    label_type: str,
) -> None:
    """
    save label per image
    """
    # Initialize a blank label image
    # For semantic segmentation, one can create a single lbl image with same label
    # For instance segmentation, one can create a single lbl image with different values per instance

    # We will create a single label
    # logging.info(f"file_name: {file_name}")
    instance_seg_lbl = np.zeros((height, width), dtype=np.uint8)
    sem_seg_lbl = np.zeros((height, width), dtype=np.uint8)

    for ann_index, ann in enumerate(annotations):
        ann_segmentation = ann["segmentation"]
        if ann_segmentation:
            # Polygons can be stored as list of coordinates
            if isinstance(ann_segmentation, list):
                # In COCO format, `segmentation` for polygons is a list of lists of coordinates
                for poly in ann_segmentation:
                    # Convert polygon to binary label
                    rles = mask_util.frPyObjects([poly], height, width)
                    sem_seg_lbl_this_instance = mask_util.decode(rles)
                    sem_seg_lbl_this_instance = np.squeeze(sem_seg_lbl_this_instance)
                    logging.info(
                        f"file_name: {file_name}, ann_index: {ann_index}, height: {height}, width: {width}, num pixels: {np.sum(sem_seg_lbl_this_instance)}, {sem_seg_lbl.shape}"
                    )
                    # Add instance to the final label.
                    # Use a unique value for each instance for multi-class labels.

                    sem_seg_lbl += sem_seg_lbl_this_instance
                    instance_seg_lbl[sem_seg_lbl_this_instance > 0] = ann_index
            # RLE format is also possible in COCO
            elif isinstance(ann_segmentation, dict):
                sem_seg_lbl_this_instance = mask_util.decode(ann_segmentation)

                sem_seg_lbl += sem_seg_lbl_this_instance
                instance_seg_lbl[sem_seg_lbl_this_instance > 0] = ann_index
            else:
                logging.info("Unknown instance for the annotation")

    # Save the generated label
    if label_type == "semantic_seg":
        sem_seg_lbl = sem_seg_lbl > 0
        sem_seg_lbl = sem_seg_lbl.astype(np.uint8)
        lbl_image = Image.fromarray(sem_seg_lbl, mode="L")
    else:
        lbl_image = Image.fromarray(instance_seg_lbl, mode="L")
    path_output_lbl_file = path_output_lbls / file_name.replace(
        ".jpeg", ".png"
    ).replace(".jpg", ".png")
    logging.info(f"output file: {path_output_lbl_file}")
    lbl_image.save(path_output_lbl_file)
    return


def hrsid_json_to_lbl_image(
    file_json: str,
    dir_output_lbls: str,
    label_type: str = "semantic_seg",
    num_workers: int = 4,
) -> None:
    """
    Converts HRSID JSON annotations into segmentation label images.

    ---------
    Arguments
    ---------
        file_json: str
            Path to the HRSID annotations JSON file.
        dir_output_lbls: str
            Directory where the generated mask images will be saved.
        label_type: str
            the type of label that needs to be saved (default: semantic_seg)

    -------
    Returns
    -------
        None
    """
    logging.info(
        f"generating lbl images for the HRSID dataset, label_type: {label_type}"
    )
    logging.info(f"input json label file: {file_json}")

    time_start = time.time()
    # Create the output directory if it doesn't exist
    path_output_lbls = Path(dir_output_lbls)
    if not path_output_lbls.is_dir():
        path_output_lbls.mkdir()
        logging.info(f"created output directory for labels: {path_output_lbls}")

    # Load the COCO-formatted HRSID JSON annotations
    with open(file_json, "r") as json_lbl_fd:
        hrsid_data = json.load(json_lbl_fd)

    # Create mappings for image and category IDs
    image_id_to_filename = {img["id"]: img["file_name"] for img in hrsid_data["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in hrsid_data["categories"]}

    # Group annotations by image
    annotations_by_image = {}
    for ann in hrsid_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    logging.info(f"Num images to be processed: {len(annotations_by_image.keys())}")

    # Process each image
    with ThreadPoolExecutor(max_workers=num_workers) as tp_executor:
        for image_id, annotations in annotations_by_image.items():
            image_info = [img for img in hrsid_data["images"] if img["id"] == image_id][
                0
            ]
            file_name = image_info["file_name"]
            height = image_info["height"]
            width = image_info["width"]

            tp_executor.submit(
                save_label_per_image,
                annotations,
                file_name,
                height,
                width,
                path_output_lbls,
                label_type,
            )

    time_end = time.time()
    logging.info(f"Total time taken: {time_end - time_start} sec.")
    return

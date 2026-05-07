import os
import math
import logging


from PIL import Image
from tqdm import tqdm
from typing import List
from tfrecord import TFRecordWriter
from pathlib import Path, PosixPath
from tfrecord.tools.tfrecord2idx import create_indices
from concurrent.futures import ProcessPoolExecutor, as_completed


def collect_file_samples(
    path_dir_images: PosixPath, file_type: str = "png"
) -> List[str]:
    """
    collect the list of files
    """
    list_file_samples = [
        f.name for f in path_dir_images.glob(f"*{file_type}") if f.is_file()
    ]

    return list_file_samples


def write_shard(
    shard_id: int,
    num_shards: int,
    list_shard_files: List[str],
    path_dir_images: PosixPath,
    path_dir_labels: PosixPath,
    path_dir_output: PosixPath,
    validate: bool,
) -> PosixPath:
    """
    function for writing one shard
    """
    path_shard = path_dir_output / f"data-{shard_id:05d}-of-{num_shards:05d}.tfrecord"

    writer = TFRecordWriter(str(path_shard))
    for file_name in list_shard_files:
        try:
            path_img = path_dir_images / file_name
            with open(path_img, "rb") as img_fd:
                img_bytes = img_fd.read()

            path_lbl = path_dir_labels / file_name
            with open(path_lbl, "rb") as lbl_fd:
                lbl_bytes = lbl_fd.read()

            if validate:
                # Optional: disable for max speed
                Image.open(path_img).verify()
                Image.open(path_lbl).verify()

            writer.write(
                {
                    "image": (img_bytes, "byte"),
                    "label": (lbl_bytes, "byte"),
                    "file_name": (os.path.basename(path_img).encode(), "byte"),
                }
            )

        except Exception as e:
            # Keep going on bad files
            logging.info(f"[Shard {shard_id}] Skipping {path_img} {path_lbl}: {e}")

    writer.close()
    return str(path_shard)


def create_tfrecords(
    dir_images: str,
    dir_labels: str,
    dir_output: str,
    num_shards: int = 16,
    num_workers: int = 8,
    validate: bool = False,
) -> None:
    """
    main pipeline to create sharded tfrecords using parallel processing
    """
    path_dir_images = Path(dir_images)
    path_dir_labels = Path(dir_labels)
    path_dir_output = Path(dir_output)

    if not path_dir_output.is_dir():
        logging.info(
            f"created directory to save the output tfrecords in {path_dir_output}"
        )
        path_dir_output.mkdir()

    list_file_samples = collect_file_samples(path_dir_images)

    num_samples = len(list_file_samples)
    logging.info(f"Total samples: {num_samples}")

    if num_workers is None:
        num_workers = min(num_samples, num_shards)

    logging.info(f"Workers: {num_workers}, Shards: {num_shards}")

    samples_per_shard = math.ceil(num_samples / num_shards)

    list_shard_file_splits = [
        list_file_samples[i * samples_per_shard : (i + 1) * samples_per_shard]
        for i in range(num_shards)
    ]

    logging.info(f"Num shard splits: {len(list_shard_file_splits)}")

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as pp_executor:
        for shard_id in range(1, num_shards + 1):
            futures.append(
                pp_executor.submit(
                    write_shard,
                    shard_id,
                    num_shards,
                    list_shard_file_splits[shard_id - 1],
                    path_dir_images,
                    path_dir_labels,
                    path_dir_output,
                    validate,
                )
            )

        # Progress tracking
        for f in tqdm(as_completed(futures), total=len(futures), desc="Shards done"):
            results.append(f.result())

    create_indices(dir_output)

    logging.info("\nFinished writing TFRecords")
    logging.info("\nOutput shards:")
    for r in sorted(results):
        logging.info(r)
    return

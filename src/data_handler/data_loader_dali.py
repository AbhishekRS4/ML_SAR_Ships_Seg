import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrecord
from nvidia.dali import backend as dali_backend


from pathlib import Path, PosixPath
from typing import Tuple, List, Union
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


dali_backend.SetHostBufferShrinkThreshold(1.0)


# ---------------------------------------------------------
# DALI Pipeline
# ---------------------------------------------------------
@pipeline_def(
    enable_conditionals=False,
    exec_pipelined=True,
    exec_async=True,
    prefetch_queue_depth=2,
)
def tfrecords_segmentation_pipeline(
    list_tfrecord_files: List[PosixPath],
    list_tfrecord_idx_files: List[PosixPath],
    shard_id: int = 0,
    num_shards: int = 1,
    random_shuffle: bool = True,
    is_train: bool = True,
    img_height: int = 800,
    img_width: int = 800,
):
    """
    Dali TFRecords pipeline
    """
    features = {
        "image": tfrecord.FixedLenFeature([], tfrecord.string, ""),
        "label": tfrecord.FixedLenFeature([], tfrecord.string, ""),
    }

    inputs = fn.readers.tfrecord(
        path=list_tfrecord_files,
        index_path=list_tfrecord_idx_files,
        features=features,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        prefetch_queue_depth=2,
        lazy_init=False,
        initial_fill=4096,
        name="TFRecordReader",
    )

    # ---------------------------------------------------------
    # Decode image + label
    # ---------------------------------------------------------
    images = fn.decoders.image(
        inputs["image"],
        device="mixed",
        output_type=types.GRAY,
        hw_decoder_load=0.90,
        preallocate_height_hint=img_height,
        preallocate_width_hint=img_width,
        bytes_per_sample_hint=img_width * img_height,
    )

    labels = fn.decoders.image(
        inputs["label"],
        device="cpu",
        output_type=types.GRAY,
    )
    labels = labels.gpu()

    # ---------------------------------------------------------
    # Optional augmentations
    # ---------------------------------------------------------
    if is_train:
        h_flip_coin = fn.random.coin_flip(probability=0.5)
        v_flip_coin = fn.random.coin_flip(probability=0.5)

        images = fn.flip(images, horizontal=h_flip_coin, vertical=v_flip_coin)
        labels = fn.flip(labels, horizontal=h_flip_coin, vertical=v_flip_coin)

    # ---------------------------------------------------------
    # Normalize image
    # ---------------------------------------------------------
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.0],
        std=[255.0],
    )

    # Label -> int64 tensor, HWC -> CHW, then squeeze channel dim
    labels = fn.cast(labels, dtype=types.INT64)
    labels = fn.squeeze(labels, axes=2)

    return images, labels


# ---------------------------------------------------------
# Build DALI Loader
# ---------------------------------------------------------
def build_dali_tfrecords_loader(
    dir_tfrecord: Union[str, PosixPath],
    batch_size: int = 8,
    num_threads: int = 4,
    device_id: int = 0,
    is_train: bool = True,
    shuffle: bool = True,
) -> DALIGenericIterator:
    """
    DALI TFRecords dataloader
    """
    path_dir_tfrecord = (
        Path(dir_tfrecord) if isinstance(dir_tfrecord, str) else dir_tfrecord
    )

    list_tfrecord_files = sorted(
        [f for f in path_dir_tfrecord.glob("*.tfrecord") if f.is_file()]
    )
    list_tfrecord_idx_files = [f.with_suffix(".tfindex") for f in list_tfrecord_files]

    pipe = tfrecords_segmentation_pipeline(
        list_tfrecord_files,
        list_tfrecord_idx_files,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        random_shuffle=shuffle,
        is_train=is_train,
    )

    pipe.build()

    data_loader = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        reader_name="TFRecordReader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=True,
        dynamic_shape=False,
    )

    return data_loader


def get_tfrecords_dataloaders(
    dir_train_tfrecords: Union[str, PosixPath],
    dir_test_tfrecords: Union[str, PosixPath],
    batch_size: int = 8,
    num_threads: int = 4,
    device_id: int = 0,
) -> Tuple[DALIGenericIterator, DALIGenericIterator]:
    """
    get TFRecords train and test dataloaders for training
    """
    train_loader = build_dali_tfrecords_loader(
        dir_train_tfrecords,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_train=True,
    )
    test_loader = build_dali_tfrecords_loader(
        dir_test_tfrecords,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_train=False,
        shuffle=False,
    )

    return train_loader, test_loader


@pipeline_def(
    enable_conditionals=False,
    exec_pipelined=True,
    exec_async=True,
    prefetch_queue_depth=2,
)
def png_segmentation_pipeline(
    list_image_files: List[str],
    list_label_files: List[str],
    shard_id: int = 0,
    num_shards: int = 1,
    random_shuffle: bool = True,
    is_train: bool = True,
    img_height: int = 800,
    img_width: int = 800,
):
    """
    DALI PNG pipeline for semantic segmentation.

    Uses paired file lists to ensure image-label correspondence is maintained
    even when shuffling is enabled.
    """
    images_encoded, _ = fn.readers.file(
        files=list_image_files,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="PNGReader",
        seed=42,
    )
    labels_encoded, _ = fn.readers.file(
        files=list_label_files,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="PNGReaderLabels",
        seed=42,
    )

    # ---------------------------------------------------------
    # Decode image + label
    # ---------------------------------------------------------
    images = fn.decoders.image(
        images_encoded,
        device="mixed",
        output_type=types.GRAY,
        hw_decoder_load=0.90,
        preallocate_height_hint=img_height,
        preallocate_width_hint=img_width,
        bytes_per_sample_hint=img_width * img_height,
    )

    labels = fn.decoders.image(
        labels_encoded,
        device="cpu",
        output_type=types.GRAY,
    )
    labels = labels.gpu()

    # ---------------------------------------------------------
    # Optional augmentations
    # ---------------------------------------------------------
    if is_train:
        h_flip_coin = fn.random.coin_flip(probability=0.5)
        v_flip_coin = fn.random.coin_flip(probability=0.5)

        images = fn.flip(images, horizontal=h_flip_coin, vertical=v_flip_coin)
        labels = fn.flip(labels, horizontal=h_flip_coin, vertical=v_flip_coin)

    # ---------------------------------------------------------
    # Normalize image
    # ---------------------------------------------------------
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.0],
        std=[255.0],
    )

    # Label -> int64 tensor, squeeze channel dim (H, W, 1) -> (H, W)
    labels = fn.cast(labels, dtype=types.INT64)
    labels = fn.squeeze(labels, axes=2)

    return images, labels


# ---------------------------------------------------------
# Build DALI Loader
# ---------------------------------------------------------
def build_dali_png_loader(
    dir_images: Union[str, PosixPath],
    dir_labels: Union[str, PosixPath],
    batch_size: int = 8,
    num_threads: int = 4,
    device_id: int = 0,
    is_train: bool = True,
    shuffle: bool = True,
) -> DALIGenericIterator:
    """
    DALI PNG dataloader
    """
    path_dir_images = Path(dir_images) if isinstance(dir_images, str) else dir_images
    path_dir_labels = Path(dir_labels) if isinstance(dir_labels, str) else dir_labels

    # Build paired file lists sorted by name to ensure correspondence
    list_image_files = sorted(
        [str(f) for f in path_dir_images.glob("*.png") if f.is_file()]
    )
    list_label_files = [str(path_dir_labels / Path(f).name) for f in list_image_files]

    pipe = png_segmentation_pipeline(
        list_image_files,
        list_label_files,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        random_shuffle=shuffle,
        is_train=is_train,
    )

    pipe.build()

    data_loader = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        reader_name="PNGReader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=True,
        dynamic_shape=False,
    )

    return data_loader


def get_png_dataloaders(
    dir_train_images: Union[str, PosixPath],
    dir_train_labels: Union[str, PosixPath],
    dir_test_images: Union[str, PosixPath],
    dir_test_labels: Union[str, PosixPath],
    batch_size: int = 8,
    num_threads: int = 4,
    device_id: int = 0,
) -> Tuple[DALIGenericIterator, DALIGenericIterator]:
    """
    get PNG train and test dataloaders for training
    """
    train_loader = build_dali_png_loader(
        dir_train_images,
        dir_train_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_train=True,
    )
    test_loader = build_dali_png_loader(
        dir_test_images,
        dir_test_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_train=False,
        shuffle=False,
    )

    return train_loader, test_loader

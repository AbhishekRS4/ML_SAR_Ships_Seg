import torch
import logging
import numpy as np
import torchvision.transforms as transforms

from skimage.io import imread
from pathlib import Path, PosixPath
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader


from data_handler.data_processing import preprocess_image_tensor


class HRSIDSemSegDataset(Dataset):
    def __init__(
        self,
        path_dir_images: PosixPath,
        path_dir_labels: PosixPath,
        which_set: str = "train",
    ):
        """
        HRSIDDataset class to load satellite image dataset in zarr format for multiple timesteps

        ----------
        Attributes
        ----------
        path_dir_images : PosixPath
            valid full directory path of the dataset with images
        path_dir_labels: PosixPath
            valid full directory path of the dataset with labels
        which_set : str
            string indicates which set to be loaded (options = ["train", "validation"])
        """
        self.path_dir_images = path_dir_images
        self.path_dir_labels = path_dir_labels

        self.list_files = sorted(
            [f.name for f in path_dir_images.glob("*png") if f.is_file()]
        )

        self.which_set = which_set

        self._affine_transform = None

        if self.which_set == "train":
            self._affine_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )

    def __len__(self):
        """
        -------
        Returns
        -------
        length : int
            number of images in the dataset list
        """
        return len(self.list_files)

    def __getitem__(self, idx):
        """
        ---------
        Arguments
        ---------
        idx : int
            index of the file

        -------
        Returns
        -------
        (image, label) : tuple of torch tensors
            tuple of normalized image and label torch tensors
        """
        # get the full file path
        file_image = self.path_dir_images / self.list_files[idx]
        file_label = self.path_dir_labels / self.list_files[idx]

        # load the image and convert it to tensor and preprocess it
        image_arr = imread(file_image)
        image_tensor = torch.from_numpy(image_arr[:, :, 0])
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = preprocess_image_tensor(image_tensor)
        # 1 x H x W

        # load label and convert it to tensor
        label_arr = imread(file_label)
        label_tensor = torch.from_numpy(label_arr)
        label_tensor = torch.unsqueeze(label_tensor, dim=0)
        # 1 x H x W

        # apply augmentation
        if self.which_set == "train":
            stacked = torch.cat([image_tensor, label_tensor], dim=0)
            # (1 + 1) x H x W
            stacked_transformed = self._affine_transform(stacked)
            # (1 + 1) x H x W

            input_image_tensor = stacked_transformed[0, :, :]
            # H x W
            input_label_tensor = stacked_transformed[1, :, :]
            # H x W

            input_image_tensor = torch.unsqueeze(input_image_tensor, dim=0)
            # 1 x H x W
            input_label_tensor = torch.unsqueeze(input_label_tensor, dim=0)
            # 1 x H x W
        else:
            input_image_tensor = image_tensor
            input_label_tensor = label_tensor
        return input_image_tensor, input_label_tensor


def get_dataloaders_for_training(
    dir_train_images: str,
    dir_train_labels: str,
    dir_test_images: str,
    dir_test_labels: str,
    batch_size: int = 32,
    num_workers: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    return the train and test dataloaders
    to be used in the training pipeline
    """
    path_dir_train_images = Path(dir_train_images)
    path_dir_train_labels = Path(dir_train_labels)
    logging.info(f"num train images: {len(path_dir_train_images.glob('*png'))}")

    path_dir_test_images = Path(dir_test_images)
    path_dir_test_labels = Path(dir_test_labels)
    logging.info(f"num test images: {len(path_dir_test_images.glob('*png'))}")

    train_set = HRSIDSemSegDataset(
        path_dir_train_images,
        path_dir_train_labels,
        which_set="train",
    )
    test_set = HRSIDSemSegDataset(
        path_dir_test_images,
        path_dir_test_labels,
        which_set="test",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader

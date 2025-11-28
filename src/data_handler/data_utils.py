import torch


def preprocess_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    preprocess the image to return normalized image in the range [0, 1]

    ---------
    Arguments
    ---------
    image_tensor: torch.Tensor
        input is a torch tensor

    -------
    Returns
    -------
    image_tensor: torch.Tensor
        returns an output torch tensor that is preprocessed
    """
    image_tensor = image_tensor / 255.0
    return image_tensor

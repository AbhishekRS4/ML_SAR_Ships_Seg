import torch
import logging
import torch.nn as nn


from typing import Union
from torch.nn import BatchNorm2d, GroupNorm, InstanceNorm2d
from torch.nn.modules.activation import ReLU, ELU, SELU, GELU, SiLU, CELU, Mish


def get_activation_func(
    activation: str = "gelu",
) -> Union[ReLU, ELU, SELU, GELU, SiLU, CELU, Mish]:
    """
    return the activation function
    """
    activation_func = None

    if activation == "relu":
        activation_func = nn.ReLU(inplace=True)
    elif activation == "elu":
        activation_func = nn.ELU(inplace=True)
    elif activation == "selu":
        activation_func = nn.SELU(inplace=True)
    elif activation == "gelu":
        activation_func = nn.GELU()
    elif activation == "silu":
        activation_func = nn.SiLU(inplace=True)
    elif activation == "celu":
        activation_func = nn.CELU(inplace=True)
    elif activation == "mish":
        activation_func = nn.Mish(inplace=True)
    else:
        logging.error(f"Unknown option for (activation={activation})")

    return activation_func

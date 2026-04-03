import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from models.resnet import resnet34
from models.model_layer_utils import get_activation_func


class UNetDecoderBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        conv_in_channels: int,
        conv_out_channels: int,
        up_in_channels: int = None,
        up_out_channels: int = None,
        activation: str = "gelu",
        norm_layer: str = "none",
    ):
        super().__init__()
        """
        eg:
        decoder1:
        up_in_channels      : 1024,     up_out_channels     : 512
        conv_in_channels    : 1024,     conv_out_channels   : 512

        decoder5:
        up_in_channels      : 64,       up_out_channels     : 64
        conv_in_channels    : 128,      conv_out_channels   : 64
        """

        self.activation_func = get_activation_func(activation=activation)

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        if up_in_channels == None:
            up_in_channels = conv_in_channels
        if up_out_channels == None:
            up_out_channels = conv_out_channels

        self.up = nn.ConvTranspose2d(
            up_in_channels, up_out_channels, kernel_size=2, stride=2
        )

        if norm_layer == "none":
            self.conv = nn.Sequential(
                nn.Conv2d(
                    conv_in_channels,
                    conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                self.activation_func,
                nn.Conv2d(
                    conv_out_channels,
                    conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                self.activation_func,
            )
        else:
            self.norm1 = self._norm_layer(conv_out_channels)
            self.norm2 = self._norm_layer(conv_out_channels)

            self.conv = nn.Sequential(
                nn.Conv2d(
                    conv_in_channels,
                    conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                self.norm1,
                self.activation_func,
                nn.Conv2d(
                    conv_out_channels,
                    conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                self.norm2,
                self.activation_func,
            )
        self.init_weights()

    def init_weights(
        self,
    ) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    # x1-upconv , x2-downconv
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

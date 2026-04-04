import torch
import torch.nn as nn
from typing import List, Union
from torch.nn import functional as F


from models.resnet import resnet34
from models.u_net import UNetDecoderBlock
from models.psa_resnet import psa_resnet34
from models.deeplab_v3_plus import DeepLabV3Plus
from models.model_layer_utils import get_activation_func
from models.convnext_v2 import convnext_v2_tiny, convnext_v2_base


class ConvNextV2TinyDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_classes: int,
        encoder_out_channels: int = 768,
        encoder_projection_in_channels: int = 96,
    ):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        """
        super().__init__()

        self.encoder = convnext_v2_tiny(num_in_channels=num_in_channels)
        self.decoder_segmenter = DeepLabV3Plus(
            encoder_out_channels,
            encoder_projection_in_channels,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]

        x = self.encoder.downsample_layers[0](x)
        encoder_stem_features = x
        x = self.encoder.stages[0](x)

        for i in range(1, 4):
            x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)

        x = self.encoder.norm(x)
        encoder_out_features = x

        x = self.decoder_segmenter(encoder_out_features, encoder_stem_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ConvNextV2BaseDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_classes: int,
        encoder_out_channels: int = 1024,
        encoder_projection_in_channels: int = 128,
    ):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        """
        super().__init__()

        self.encoder = convnext_v2_base(num_in_channels=num_in_channels)
        self.decoder_segmenter = DeepLabV3Plus(
            encoder_out_channels,
            encoder_projection_in_channels,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]

        x = self.encoder.downsample_layers[0](x)
        encoder_stem_features = x
        x = self.encoder.stages[0](x)

        for i in range(1, 4):
            x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)

        x = self.encoder.norm(x)
        encoder_out_features = x

        x = self.decoder_segmenter(encoder_out_features, encoder_stem_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet34UNet(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_classes: int,
        activation: str = "gelu",
        norm_layer: Union[str, None] = None,
        dropout_rate: float = 0.2,
        filters: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.activation_func = get_activation_func(activation=activation)
        resnet34_encoder = resnet34(
            num_in_channels,
            activation=activation,
            filters=filters,
        )
        self.filters = filters

        self.first_layer = nn.Sequential(*list(resnet34_encoder.children())[:3])

        self.encoder1 = resnet34_encoder.layer1
        self.encoder2 = resnet34_encoder.layer2
        self.encoder3 = resnet34_encoder.layer3
        self.encoder4 = resnet34_encoder.layer4

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.bridge_norm = self._norm_layer(self.filters[3] * 2)
        self.bridge = nn.Sequential(
            nn.Conv2d(
                self.filters[3],
                self.filters[3] * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            self.bridge_norm,
            self.activation_func,
        )

        self.decoder1 = UNetDecoderBlock(
            conv_in_channels=self.filters[3] * 2,
            conv_out_channels=self.filters[3],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder2 = UNetDecoderBlock(
            conv_in_channels=self.filters[2] * 2,
            conv_out_channels=self.filters[2],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder3 = UNetDecoderBlock(
            conv_in_channels=self.filters[1] * 2,
            conv_out_channels=self.filters[1],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder4 = UNetDecoderBlock(
            conv_in_channels=self.filters[0] * 2,
            conv_out_channels=self.filters[0],
            activation=activation,
            norm_layer=norm_layer,
        )

        if dropout_rate != 0:
            self.last_layer = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.ConvTranspose2d(
                    in_channels=self.filters[0],
                    out_channels=self.filters[0],
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.filters[0], num_classes, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            self.last_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.filters[0],
                    out_channels=self.filters[0],
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.filters[0], num_classes, kernel_size=3, padding=1, bias=False
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.first_layer(x)
        # 64, H/2, W/2
        e2 = self.encoder1(e1)
        # 64, H/2, W/2
        e3 = self.encoder2(e2)
        # 128, H/4, W/4
        e4 = self.encoder3(e3)
        # 256, H/8, W/8
        e5 = self.encoder4(e4)
        # 512, H/16, W/16

        c = self.bridge(e5)
        # 512, H/32, W/32

        d1 = self.decoder1(c, e5)
        # 512, H/16, W/16
        d2 = self.decoder2(d1, e4)
        # 256, H/8, W/8
        d3 = self.decoder3(d2, e3)
        # 128, H/4, W/4
        d4 = self.decoder4(d3, e2)
        # 64, H/2, W/2

        out = self.last_layer(d4)
        # C, H, W

        return out


class PSAResNet34UNet(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_classes: int,
        activation: str = "gelu",
        norm_layer: Union[str, None] = None,
        dropout_rate: float = 0.2,
        filters: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.activation_func = get_activation_func(activation=activation)
        psa_resnet34_encoder = psa_resnet34(
            num_in_channels,
            activation=activation,
            filters=filters,
        )
        self.filters = filters

        self.first_layer = nn.Sequential(*list(psa_resnet34_encoder.children())[:3])

        self.encoder1 = psa_resnet34_encoder.layer1
        self.encoder2 = psa_resnet34_encoder.layer2
        self.encoder3 = psa_resnet34_encoder.layer3
        self.encoder4 = psa_resnet34_encoder.layer4

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.bridge_norm = self._norm_layer(self.filters[3] * 2)
        self.bridge = nn.Sequential(
            nn.Conv2d(
                self.filters[3],
                self.filters[3] * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            self.bridge_norm,
            self.activation_func,
        )

        self.decoder1 = UNetDecoderBlock(
            conv_in_channels=self.filters[3] * 2,
            conv_out_channels=self.filters[3],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder2 = UNetDecoderBlock(
            conv_in_channels=self.filters[2] * 2,
            conv_out_channels=self.filters[2],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder3 = UNetDecoderBlock(
            conv_in_channels=self.filters[1] * 2,
            conv_out_channels=self.filters[1],
            activation=activation,
            norm_layer=norm_layer,
        )
        self.decoder4 = UNetDecoderBlock(
            conv_in_channels=self.filters[0] * 2,
            conv_out_channels=self.filters[0],
            activation=activation,
            norm_layer=norm_layer,
        )

        if dropout_rate != 0:
            self.last_layer = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.ConvTranspose2d(
                    in_channels=self.filters[0],
                    out_channels=self.filters[0],
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.filters[0], num_classes, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            self.last_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.filters[0],
                    out_channels=self.filters[0],
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.filters[0], num_classes, kernel_size=3, padding=1, bias=False
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.first_layer(x)
        # 64, H/2, W/2
        e2 = self.encoder1(e1)
        # 64, H/2, W/2
        e3 = self.encoder2(e2)
        # 128, H/4, W/4
        e4 = self.encoder3(e3)
        # 256, H/8, W/8
        e5 = self.encoder4(e4)
        # 512, H/16, W/16

        c = self.bridge(e5)
        # 512, H/32, W/32

        d1 = self.decoder1(c, e5)
        # 512, H/16, W/16
        d2 = self.decoder2(d1, e4)
        # 256, H/8, W/8
        d3 = self.decoder3(d2, e3)
        # 128, H/4, W/4
        d4 = self.decoder4(d3, e2)
        # 64, H/2, W/2

        out = self.last_layer(d4)
        # C, H, W

        return out

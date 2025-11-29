import torch
import torch.nn as nn
from torch.nn import functional as F


from models.deeplab_v3_plus import DeepLabV3Plus
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

import torch
import torch.nn as nn
from torch.nn import functional as F


from models.deep_lab_v3_plus import DeepLabV3Plus
from models.conv_next_v2 import convnextv2_tiny, convnextv2_base


class ConvNextV2TinyDeepLabV3Plus(nn.Module):
    def __init__(
        self,
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

        self.encoder = convnextv2_tiny()
        self.decoder_segmenter = DeepLabV3Plus(
            encoder_out_channels,
            encoder_projection_in_channels,
            num_classes,
        )

    def forward(self, x):
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
        num_classes,
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

        self.encoder = convnextv2_base()
        self.decoder_segmenter = DeepLabV3Plus(
            encoder_out_channels,
            encoder_projection_in_channels,
            num_classes,
        )

    def forward(self, x):
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

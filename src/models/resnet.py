import torch
import torch.nn as nn


from torch import Tensor
from typing import List, Union, Callable, Optional
from models.model_layer_utils import get_activation_func
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        activation: str = "gelu",
        norm_layer: str = "none",
    ) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.activation_func = get_activation_func(activation=activation)

        if norm_layer == "none":
            self.conv1 = nn.Sequential(
                conv3x3(inplanes, planes, stride),
            )

            self.conv2 = nn.Sequential(
                conv3x3(planes, planes),
            )
        else:
            norm_layer = nn.BatchNorm2d
            self.norm1 = norm_layer(planes)
            self.norm2 = norm_layer(planes)

            self.conv1 = nn.Sequential(
                conv3x3(inplanes, planes, stride),
                self.norm1,
            )
            self.conv2 = nn.Sequential(
                conv3x3(planes, planes),
                self.norm2,
            )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.activation_func(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation_func(out)

        return out


class CustomResNet(nn.Module):
    def __init__(
        self,
        num_input_feats: int,
        layers: List[int],
        block=Union[BasicBlock, Bottleneck],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Union[List, None] = None,
        norm_layer: Union[nn.BatchNorm2d, None] = None,
        activation: str = "gelu",
        filters: List[int] = [64, 128, 256, 512],
    ):
        """
        CustomResNet class to build the CustomResNet encoder model

        ----------
        Attributes
        ----------
        num_input_feats: int
            number of input features
        layers : list
            list of number of layers in each residual block
        block : Union[BasicBlock, Bottleneck]
            type of the residual block (options = [BasicBlock, Bottleneck])
        zero_init_residual : bool
            to indicate whether to use zero weights for BN
        groups : int
            indicates the number of groups (default: 1)
        width_per_group : int
            indicates the width per group (default: 64)
        replace_stride_with_dilation : list
            a list indicating whether to replace stride with dilation (default: None)
        norm_layer : object
            object of type batch norm (default: None)
        """

        super(CustomResNet, self).__init__()

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.filters = filters
        self.inplanes = self.filters[0]
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            num_input_feats,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.activation_func = get_activation_func(activation=activation)

        self.layer1 = self._make_layer(
            block,
            self.filters[0],
            layers[0],
            activation=activation,
        )
        self.layer2 = self._make_layer(
            block,
            self.filters[1],
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            activation=activation,
        )
        self.layer3 = self._make_layer(
            block,
            self.filters[2],
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            activation=activation,
        )
        self.layer4 = self._make_layer(
            block,
            self.filters[3],
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            activation=activation,
        )

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        activation: str = "gelu",
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                activation=activation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ---------
        Arguments
        ---------
        x : torch tensor
            a tensor of input features

        -------
        Returns
        -------
        x : torch tensor
            output of the CustomResNet
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_func(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(
    num_input_feats: int,
    block_type: Union[BasicBlock, Bottleneck],
    layers: List[int],
    activation: str = "gelu",
    filters: List[int] = [64, 128, 256, 512],
) -> CustomResNet:
    """
    ---------
    Arguments
    ---------
    block_type : Union[BasicBlock]
        object of type block
    layers : list
        list of layers in each residual block

    -------
    Returns
    -------
    model : CustomResNet
        model object of type CustomResNet
    """
    model = CustomResNet(
        num_input_feats, layers, block_type, activation=activation, filters=filters
    )
    return model


def resnet34(
    num_input_feats: int,
    activation: str = "gelu",
    filters: List[int] = [64, 128, 256, 512],
) -> CustomResNet:
    r"""
    ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(
        num_input_feats,
        BasicBlock,
        [3, 4, 6, 3],
        activation=activation,
        filters=filters,
    )

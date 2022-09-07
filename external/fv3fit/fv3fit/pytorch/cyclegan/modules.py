import logging

from typing import Callable, Literal, Protocol, Union
import torch.nn as nn

logger = logging.getLogger(__name__)


def relu_activation(**kwargs):
    def relu_factory():
        return nn.ReLU(**kwargs)

    return relu_factory


def tanh_activation():
    return nn.Tanh()


def leakyrelu_activation(**kwargs):
    def leakyrelu_factory():
        return nn.LeakyReLU(**kwargs)

    return leakyrelu_factory


def no_activation():
    return nn.Identity()


class ConvolutionFactory(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[str, int] = 0,
        output_padding: int = 0,
        stride: int = 1,
        stride_type: Literal["regular", "transpose"] = "regular",
        bias: bool = True,
    ) -> nn.Module:
        """
        Create a convolutional layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of the convolution kernel
            padding: padding to apply to the input, should be an integer or "same"
            output_padding: argument used for transpose convolution
            stride: stride of the convolution
            stride_type: type of stride, one of "regular" or "transpose"
            bias: whether to include a bias vector in the produced layers
        """
        ...


class CurriedModuleFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a torch module.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
        """
        ...


def single_tile_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: Union[str, int] = 0,
    output_padding: int = 0,
    stride: int = 1,
    stride_type: Literal["regular", "transpose"] = "regular",
    bias: bool = True,
) -> ConvolutionFactory:
    """
    Construct a convolutional layer for single tile data (like images).

    Args:
        kernel_size: size of the convolution kernel
        padding: padding to apply to the input, should be an integer or "same"
        output_padding: argument used for transpose convolution
        stride: stride of the convolution
        stride_type: type of stride, one of "regular" or "transpose"
        bias: whether to include a bias vector in the produced layers
    """
    if stride == 1:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    elif stride_type == "regular":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif stride_type == "transpose":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(padding, padding),
            output_padding=output_padding,
            bias=bias,
        )


class ResnetBlock(nn.Module):
    """
    Residual network block as defined in He et al. 2016,
    https://arxiv.org/abs/1512.03385.

    Contains two convolutional layers with instance normalization, and an
    activation function applied to the first layer's instance-normalized output.
    The input to the block is added to the output of the final convolutional layer.
    """

    def __init__(
        self,
        channels: int,
        convolution_factory: CurriedModuleFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation(),
    ):
        """
        Args:
            channels: number of input channels and filters in the convolutional layers
            convolution_factory: factory for creating convolutional layers
            activation_factory: factory for creating activation layers
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                convolution_factory=convolution_factory,
                activation_factory=activation_factory,
            ),
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                convolution_factory=convolution_factory,
                activation_factory=no_activation,
            ),
        )
        self.identity = nn.Identity()

    def forward(self, inputs):
        g = self.conv_block(inputs)
        return g + self.identity(inputs)


class ConvBlock(nn.Module):
    """
    Module packaging a convolutional layer with instance normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convolution_factory: CurriedModuleFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation(),
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            convolution_factory: factory for creating convolutional layers
            activation_factory: factory for creating activation layers
        """
        super(ConvBlock, self).__init__()
        # it's helpful to package this code into a class so that we can e.g. see what
        # happens when globally disabling InstanceNorm2d or switching to another type
        # of normalization, while debugging.
        self.conv_block = nn.Sequential(
            convolution_factory(in_channels=in_channels, out_channels=out_channels),
            nn.InstanceNorm2d(out_channels),
            activation_factory(),
        )

    def forward(self, inputs):
        return self.conv_block(inputs)

from typing import Callable, Literal, Protocol, Union
import torch
import torch.nn as nn
from toolz import curry


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


class CurriedConvolutionFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int,) -> nn.Module:
        """
        Create a convolutional layer.

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
    def __init__(
        self,
        n_filters: int,
        convolution_factory: CurriedConvolutionFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation(),
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=n_filters,
                out_channels=n_filters,
                convolution_factory=convolution_factory,
                activation_factory=activation_factory,
            ),
            ConvBlock(
                in_channels=n_filters,
                out_channels=n_filters,
                convolution_factory=convolution_factory,
                activation_factory=no_activation,
            ),
        )
        self.identity = nn.Identity()

    def forward(self, inputs):
        g = self.conv_block(inputs)
        return g + self.identity(inputs)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convolution_factory: CurriedConvolutionFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation(),
    ):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            convolution_factory(in_channels=in_channels, out_channels=out_channels),
            nn.InstanceNorm2d(out_channels),
            activation_factory(),
        )

    def forward(self, inputs):
        return self.conv_block(inputs)


class Generator(nn.Module):
    def __init__(
        self,
        channels: int,
        n_convolutions: int,
        n_resnet: int,
        max_filters: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        super(Generator, self).__init__()

        def resnet(in_channels: int):
            resnet_blocks = [
                ResnetBlock(
                    n_filters=in_channels,
                    convolution_factory=curry(convolution)(
                        kernel_size=3, padding="same"
                    ),
                    activation_factory=relu_activation(),
                )
                for _ in range(n_resnet)
            ]
            return nn.Sequential(*resnet_blocks)

        def down(in_channels: int, out_channels: int):
            return ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                convolution_factory=curry(convolution)(
                    kernel_size=3, stride=2, padding=1
                ),
                activation_factory=relu_activation(),
            )

        def up(in_channels: int, out_channels: int):
            return ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                convolution_factory=curry(convolution)(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    stride_type="transpose",
                ),
                activation_factory=relu_activation(),
            )

        min_filters = int(max_filters / 2 ** (n_convolutions - 1))

        self._first_conv = nn.Sequential(
            convolution(
                kernel_size=3,
                padding="same",
                in_channels=channels,
                out_channels=min_filters,
            ),
            relu_activation()(),
        )

        self._unet = UNet(
            down_factory=down,
            up_factory=up,
            bottom_factory=resnet,
            depth=n_convolutions - 1,
            in_channels=min_filters,
        )

        self._out_conv = convolution(
            kernel_size=3,
            padding="same",
            in_channels=2 * min_filters,
            out_channels=channels,
        )

    def forward(self, inputs):
        # data will have channels last, model requires channels first
        inputs = inputs.permute(0, 3, 1, 2)
        x = self._first_conv(inputs)
        x = self._unet(x)
        outputs = self._out_conv(x)
        return outputs.permute(0, 2, 3, 1)


class UNet(nn.Module):
    def __init__(
        self, down_factory, up_factory, bottom_factory, depth: int, in_channels: int,
    ):
        super(UNet, self).__init__()
        lower_channels = 2 * in_channels
        self._down = down_factory(in_channels=in_channels, out_channels=lower_channels)
        self._up = up_factory(in_channels=lower_channels, out_channels=in_channels)
        if depth == 1:
            self._lower = bottom_factory(in_channels=lower_channels)
        elif depth <= 0:
            raise ValueError(f"depth must be at least 1, got {depth}")
        else:
            self._lower = UNet(
                down_factory,
                up_factory,
                bottom_factory,
                depth=depth - 1,
                in_channels=lower_channels,
            )

    def forward(self, inputs):
        x = self._down(inputs)
        x = self._lower(x)
        x = self._up(x)
        # skip connection
        x = torch.concat([x, inputs], dim=1)
        return x

import dataclasses
from typing import Callable, Literal, Optional, Protocol
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from ..graph import build_dgl_graph, CubedSphereGraphOperation
from toolz import curry


def relu_activation():
    return nn.ReLU()


def tanh_activation():
    return nn.Tanh()


def leakyrelu_activation(**kwargs):
    def factory():
        return nn.LeakyReLU(**kwargs)

    return factory


def no_activation():
    return nn.Identity()


class ConvolutionFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int) -> nn.Module:
        ...


class ConvolutionFactoryFactory(Protocol):
    def __call__(
        self,
        kernel_size: int,
        padding: int,
        stride: int = 1,
        stride_type: Literal["regular", "transpose"] = "regular",
        bias: bool = True,
    ) -> ConvolutionFactory:
        ...


@curry
def strided_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool = True,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


@curry
def transpose_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    bias: bool = True,
):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=(padding, padding),
        output_padding=output_padding,
        bias=bias,
    )


@curry
def flat_convolution(in_channels: int, out_channels: int, kernel_size: int, bias=True):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="same",
        bias=bias,
    )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        convolution_factory: ConvolutionFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation,
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
        convolution_factory: ConvolutionFactory,
        activation_factory: Callable[[], nn.Module] = relu_activation,
    ):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            convolution_factory(in_channels=in_channels, out_channels=out_channels),
            nn.InstanceNorm2d(out_channels),
            activation_factory(),
        )

    def forward(self, inputs):
        return self.conv_block(inputs)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, n_convolutions: int, max_filters: int):
        super(Discriminator, self).__init__()
        # max_filters = min_filters * 2 ** (n_convolutions - 1), therefore
        min_filters = int(max_filters / 2 ** (n_convolutions - 1))
        convs = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=min_filters,
                convolution_factory=strided_convolution(
                    kernel_size=3, stride=2, padding=1
                ),
                activation_factory=leakyrelu_activation(alpha=0.2),
            )
        ]
        for i in range(1, n_convolutions):
            convs.append(
                ConvBlock(
                    in_channels=min_filters * 2 ** (i - 1),
                    out_channels=min_filters * 2 ** i,
                    convolution_factory=strided_convolution(
                        kernel_size=3, stride=2, padding=1
                    ),
                    activation_factory=leakyrelu_activation(alpha=0.2),
                )
            )
        final_conv = ConvBlock(
            in_channels=max_filters,
            out_channels=max_filters,
            convolution_factory=flat_convolution(kernel_size=3),
            activation_factory=leakyrelu_activation(alpha=0.2),
        )
        patch_output = ConvBlock(
            in_channels=max_filters,
            out_channels=1,
            convolution_factory=flat_convolution(kernel_size=3),
            activation_factory=leakyrelu_activation(alpha=0.2),
        )
        self._sequential = nn.Sequential(*convs, final_conv, patch_output)

    def forward(self, inputs):
        return self._sequential(inputs)


class Generator(nn.Module):
    def __init__(
        self, channels: int, n_convolutions: int, n_resnet: int, max_filters: int,
    ):
        super(Generator, self).__init__()
        min_filters = int(max_filters / 2 ** (n_convolutions - 1))
        convs = [
            ConvBlock(
                in_channels=channels,
                out_channels=min_filters,
                convolution_factory=flat_convolution(kernel_size=7),
                activation_factory=relu_activation,
            )
        ]
        for i in range(1, n_convolutions):
            convs.append(
                ConvBlock(
                    in_channels=min_filters * 2 ** (i - 1),
                    out_channels=min_filters * 2 ** i,
                    convolution_factory=strided_convolution(
                        kernel_size=3, stride=2, padding=1
                    ),
                    activation_factory=relu_activation,
                )
            )
        resnet_blocks = [
            ResnetBlock(
                n_filters=max_filters,
                convolution_factory=flat_convolution(kernel_size=3),
                activation_factory=relu_activation,
            )
            for i in range(n_resnet)
        ]
        transpose_convs = []
        for i in range(1, n_convolutions):
            transpose_convs.append(
                ConvBlock(
                    in_channels=max_filters // (2 ** (i - 1)),
                    out_channels=max_filters // (2 ** i),
                    convolution_factory=transpose_convolution(
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    activation_factory=relu_activation,
                )
            )
        out_conv = ConvBlock(
            in_channels=min_filters,
            out_channels=channels,
            convolution_factory=flat_convolution(kernel_size=7),
            activation_factory=tanh_activation,
        )
        self._sequential = nn.Sequential(
            *convs, *resnet_blocks, *transpose_convs, out_conv
        )

    def forward(self, inputs: torch.Tensor):
        # data will have channels last, model requires channels first
        inputs = inputs.permute(0, 3, 1, 2)
        outputs: torch.Tensor = self._sequential(inputs)
        return outputs.permute(0, 2, 3, 1)

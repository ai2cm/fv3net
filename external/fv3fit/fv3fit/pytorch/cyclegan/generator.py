import dataclasses
import torch.nn as nn
from toolz import curry
import torch
from .modules import (
    ConvBlock,
    ConvolutionFactory,
    single_tile_convolution,
    relu_activation,
    ResnetBlock,
)


@dataclasses.dataclass
class GeneratorConfig:
    n_convolutions: int = 3
    n_resnet: int = 3
    kernel_size: int = 3
    max_filters: int = 256

    def build(
        self, channels: int, convolution: ConvolutionFactory = single_tile_convolution,
    ):
        return Generator(
            channels=channels,
            n_convolutions=self.n_convolutions,
            n_resnet=self.n_resnet,
            kernel_size=self.kernel_size,
            max_filters=self.max_filters,
            convolution=convolution,
        )


class Generator(nn.Module):
    def __init__(
        self,
        channels: int,
        n_convolutions: int,
        n_resnet: int,
        kernel_size: int,
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
                    kernel_size=kernel_size,
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
                kernel_size=7,
                in_channels=channels,
                out_channels=min_filters,
                padding="same",
            ),
            relu_activation()(),
        )

        self._encoder_decoder = SymmetricEncoderDecoder(
            down_factory=down,
            up_factory=up,
            bottom_factory=resnet,
            depth=n_convolutions - 1,
            in_channels=min_filters,
        )

        self._out_conv = convolution(
            kernel_size=7,
            in_channels=min_filters,
            out_channels=channels,
            padding="same",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._first_conv(inputs)
        x = self._encoder_decoder(x)
        outputs: torch.Tensor = self._out_conv(x)
        return outputs


class SymmetricEncoderDecoder(nn.Module):
    """
    Encoder-decoder network with a symmetric structure.

    Not a u-net because it does not have skip connections.
    """

    def __init__(
        self, down_factory, up_factory, bottom_factory, depth: int, in_channels: int,
    ):
        super(SymmetricEncoderDecoder, self).__init__()
        lower_channels = 2 * in_channels
        self._down = down_factory(in_channels=in_channels, out_channels=lower_channels)
        self._up = up_factory(in_channels=lower_channels, out_channels=in_channels)
        if depth == 1:
            self._lower = bottom_factory(in_channels=lower_channels)
        elif depth <= 0:
            raise ValueError(f"depth must be at least 1, got {depth}")
        else:
            self._lower = SymmetricEncoderDecoder(
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
        return x

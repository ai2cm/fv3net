import dataclasses

import torch.nn as nn
from toolz import curry
import torch
from .modules import (
    ConvolutionFactory,
    single_tile_convolution,
    leakyrelu_activation,
    ConvBlock,
)


@dataclasses.dataclass
class DiscriminatorConfig:

    n_convolutions: int = 3
    kernel_size: int = 3
    max_filters: int = 256

    def build(
        self, channels: int, convolution: ConvolutionFactory = single_tile_convolution,
    ):
        return Discriminator(
            in_channels=channels,
            n_convolutions=self.n_convolutions,
            kernel_size=self.kernel_size,
            max_filters=self.max_filters,
            convolution=convolution,
        )


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_convolutions: int,
        kernel_size: int,
        max_filters: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        super(Discriminator, self).__init__()
        # max_filters = min_filters * 2 ** (n_convolutions - 1), therefore
        min_filters = int(max_filters / 2 ** (n_convolutions - 1))
        convs = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=min_filters,
                convolution_factory=curry(convolution)(
                    kernel_size=kernel_size, stride=2, padding=1
                ),
                activation_factory=leakyrelu_activation(
                    negative_slope=0.2, inplace=True
                ),
            )
        ]
        for i in range(1, n_convolutions):
            convs.append(
                ConvBlock(
                    in_channels=min_filters * 2 ** (i - 1),
                    out_channels=min_filters * 2 ** i,
                    convolution_factory=curry(convolution)(
                        kernel_size=kernel_size, stride=2, padding=1
                    ),
                    activation_factory=leakyrelu_activation(
                        negative_slope=0.2, inplace=True
                    ),
                )
            )
        final_conv = ConvBlock(
            in_channels=max_filters,
            out_channels=max_filters,
            convolution_factory=curry(convolution)(kernel_size=kernel_size),
            activation_factory=leakyrelu_activation(negative_slope=0.2, inplace=True),
        )
        patch_output = convolution(
            kernel_size=3, in_channels=max_filters, out_channels=1, padding="same"
        )
        self._sequential = nn.Sequential(*convs, final_conv, patch_output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._sequential(inputs)

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
    CurriedModuleFactory,
)


@dataclasses.dataclass
class GeneratorConfig:
    """
    Configuration for a generator network.

    Follows the architecture of Zhu et al. 2017, https://arxiv.org/abs/1703.10593.
    This network contains an initial convolutional layer with kernel size 7,
    strided convolutions with stride of 2, multiple residual blocks,
    fractionally strided convolutions with stride 1/2, followed by an output
    convolutional layer with kernel size 7 to map to the output channels.

    Attributes:
        n_convolutions: number of strided convolutional layers after the initial
            convolutional layer and before the residual blocks
        n_resnet: number of residual blocks
        kernel_size: size of convolutional kernels in the strided convolutions
            and resnet blocks
        max_filters: maximum number of filters in any convolutional layer,
            equal to the number of filters in the final strided convolutional layer
            and in the resnet blocks
    """

    n_convolutions: int = 3
    n_resnet: int = 3
    kernel_size: int = 3
    max_filters: int = 256

    def build(
        self, channels: int, convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            channels: number of input channels
            convolution: factory for creating all convolutional layers
                used by the network
        """
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
        """
        Args:
            channels: number of input and output channels
            n_convolutions: number of strided convolutional layers after the initial
                convolutional layer and before the residual blocks
            n_resnet: number of residual blocks
            kernel_size: size of convolutional kernels in the strided convolutions
                and resnet blocks
            max_filters: maximum number of filters in any convolutional layer,
                equal to the number of filters in the final strided convolutional layer
                and in the resnet blocks
            convolution: factory for creating all convolutional layers
                used by the network
        """
        super(Generator, self).__init__()

        def resnet(in_channels: int, out_channels: int):
            if in_channels != out_channels:
                raise ValueError(
                    "resnet must have same number of output channels as "
                    "input channels, since the inputs are added to the outputs"
                )
            resnet_blocks = [
                ResnetBlock(
                    channels=in_channels,
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

        min_filters = int(max_filters / 2 ** n_convolutions)

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
            depth=n_convolutions,
            in_channels=min_filters,
        )

        self._out_conv = convolution(
            kernel_size=7,
            in_channels=min_filters,
            out_channels=channels,
            padding="same",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
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
        self,
        down_factory: CurriedModuleFactory,
        up_factory: CurriedModuleFactory,
        bottom_factory: CurriedModuleFactory,
        depth: int,
        in_channels: int,
    ):
        """
        Args:
            down_factory: factory for creating a downsample module which reduces
                height and width by a factor of 2, such as strided convolution
            up_factory: factory for creating an upsample module which doubles
                height and width, such as fractionally strided convolution
            bottom_factory: factory for creating the bottom module which keeps
                height and width constant
        """
        super(SymmetricEncoderDecoder, self).__init__()
        lower_channels = 2 * in_channels
        self._down = down_factory(in_channels=in_channels, out_channels=lower_channels)
        self._up = up_factory(in_channels=lower_channels, out_channels=in_channels)
        if depth == 1:
            self._lower = bottom_factory(
                in_channels=lower_channels, out_channels=lower_channels
            )
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        x = self._down(inputs)
        x = self._lower(x)
        x = self._up(x)
        return x

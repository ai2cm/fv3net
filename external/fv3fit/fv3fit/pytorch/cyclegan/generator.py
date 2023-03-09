import dataclasses
from typing import Tuple
import torch.nn as nn
from toolz import curry
import torch
from .modules import (
    ConvBlock,
    ConvolutionFactory,
    FoldFirstDimension,
    single_tile_convolution,
    relu_activation,
    ResnetBlock,
    CurriedModuleFactory,
    GeographicFeatures,
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

    Generally we suggest using an even kernel size for strided convolutions,
    and an odd kernel size for resnet (non-strided) convolutions. For an explanation
    of this, see the docstring on halo_convolution.

    Attributes:
        n_convolutions: number of strided convolutional layers after the initial
            convolutional layer and before the residual blocks
        n_resnet: number of residual blocks
        kernel_size: size of convolutional kernels in the resnet blocks
        strided_kernel_size: size of convolutional kernels in the
            strided convolutions
        max_filters: maximum number of filters in any convolutional layer,
            equal to the number of filters in the final strided convolutional layer
            and in the resnet blocks
        use_geographic_bias: if True, include a layer that adds a trainable bias
            vector that is a function of x and y to the input and output of the network
        disable_convolutions: if True, ignore all layers other than bias (if enabled).
            Useful for debugging and for testing the effect of the
            geographic bias layer.
    """

    n_convolutions: int = 3
    n_resnet: int = 3
    kernel_size: int = 3
    strided_kernel_size: int = 4
    max_filters: int = 256
    use_geographic_bias: bool = True
    disable_convolutions: bool = False

    def build(
        self,
        channels: int,
        nx: int,
        ny: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            channels: number of input channels
            nx: number of x grid points
            ny: number of y grid points
            convolution: factory for creating all convolutional layers
                used by the network
        """
        return Generator(
            config=self, channels=channels, convolution=convolution, nx=nx, ny=ny,
        )


class GeographicBias(nn.Module):
    """
    Adds a trainable bias vector of shape [6, channels, nx, ny] to the layer input.
    """

    def __init__(self, channels: int, nx: int, ny: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6, channels, nx, ny))

    def forward(self, x):
        return x + self.bias


class Generator(nn.Module):
    def __init__(
        self,
        config: GeneratorConfig,
        channels: int,
        nx: int,
        ny: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            config: pre-defined configuration for the generator network which is not
                defined by input data or higher-level configuration
            channels: number of input and output channels
            nx: number of grid points in x direction
            ny: number of grid points in y direction
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
                        kernel_size=config.kernel_size
                    ),
                    activation_factory=relu_activation(),
                )
                for _ in range(config.n_resnet)
            ]
            return nn.Sequential(*resnet_blocks)

        if config.strided_kernel_size % 2 == 0:
            output_padding = 0
        else:
            output_padding = 1

        def down(in_channels: int, out_channels: int):
            return ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                convolution_factory=curry(convolution)(
                    kernel_size=config.strided_kernel_size,
                    stride=2,
                    output_padding=output_padding,
                ),
                activation_factory=relu_activation(),
            )

        def up(in_channels: int, out_channels: int):
            return ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                convolution_factory=curry(convolution)(
                    kernel_size=config.strided_kernel_size,
                    stride=2,
                    output_padding=output_padding,
                    stride_type="transpose",
                ),
                activation_factory=relu_activation(),
            )

        min_filters = int(config.max_filters / 2 ** config.n_convolutions)

        if config.disable_convolutions:
            main = nn.Identity()
        else:
            in_channels = channels + GeographicFeatures.N_FEATURES
            first_conv = nn.Sequential(
                convolution(
                    kernel_size=7, in_channels=in_channels, out_channels=min_filters,
                ),
                FoldFirstDimension(nn.InstanceNorm2d(min_filters)),
                relu_activation()(),
            )

            encoder_decoder = SymmetricEncoderDecoder(
                down_factory=down,
                up_factory=up,
                bottom_factory=resnet,
                depth=config.n_convolutions,
                in_channels=min_filters,
            )

            out_conv = nn.Sequential(
                convolution(
                    kernel_size=7, in_channels=min_filters, out_channels=channels,
                ),
            )
            main = nn.Sequential(first_conv, encoder_decoder, out_conv)
        self._main = main
        self._geographic_features = GeographicFeatures(nx=nx, ny=ny)
        if config.use_geographic_bias:
            self._input_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
            self._output_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
        else:
            self._input_bias = nn.Identity()
            self._output_bias = nn.Identity()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: A tuple containing a tensor of shape (batch, 1) with the time and
                a tensor of shape (batch, tile, in_channels, height, width)

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        # TODO: this is a hack to support the old API, when it can be
        # removed this should be changed to
        # time, state = inputs
        # x = self._input_bias(state)
        # x = self._geographic_features((time, x))
        # x = self._main(x)
        # outputs: torch.Tensor = self._output_bias(x)
        # return outputs
        try:
            time, state = inputs
        except ValueError:
            time, state = None, inputs
        x = self._input_bias(state)
        if time is not None and hasattr(self, "_geographic_features"):
            x = self._geographic_features((time, x))
        x = self._main(x)
        outputs: torch.Tensor = self._output_bias(x)
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

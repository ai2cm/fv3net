import dataclasses
from typing import Tuple
from fv3fit.pytorch.system import DEVICE
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
)


@dataclasses.dataclass
class RecurrentGeneratorConfig:
    """
    Configuration for a recurrent generator network.

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
        return RecurrentGenerator(
            config=self, channels=channels, nx=nx, ny=ny, convolution=convolution,
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


class ResnetHiddenWrapper(nn.Module):
    def __init__(
        self,
        channels: int,
        resnet: nn.Module,
        convolution_factory: CurriedModuleFactory,
    ):
        super().__init__()
        self.first_block = ConvBlock(
            in_channels=channels * 2,  # hidden channels will be concatenated
            out_channels=channels,
            convolution_factory=convolution_factory,
            activation_factory=relu_activation(),
        )
        self.resnet = resnet

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        x = torch.cat([x, hidden], dim=-3)
        x = self.first_block(x)
        x = self.resnet(x)
        return x, x  # the output is the hidden state


class RecurrentGenerator(nn.Module):
    def __init__(
        self,
        config: RecurrentGeneratorConfig,
        channels: int,
        nx: int,
        ny: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            config: configuration for the network
            channels: number of input and output channels
            nx: number of grid points in x direction
            ny: number of grid points in y direction
            convolution: factory for creating all convolutional layers
                used by the network
        """
        super(RecurrentGenerator, self).__init__()
        self.channels = channels
        self.hidden_channels = config.max_filters
        self.nx = nx
        self.ny = ny
        self.n_convolutions = config.n_convolutions

        def resnet(in_channels: int, out_channels: int):
            if in_channels != out_channels:
                raise ValueError(
                    "resnet must have same number of output channels as "
                    "input channels, since the inputs are added to the outputs"
                )
            convolution_factory = curry(convolution)(kernel_size=config.kernel_size)
            resnet_blocks = [
                ResnetBlock(
                    channels=in_channels,
                    convolution_factory=convolution_factory,
                    activation_factory=relu_activation(),
                )
                for _ in range(config.n_resnet)
            ]
            return ResnetHiddenWrapper(
                channels=in_channels,
                resnet=nn.Sequential(*resnet_blocks),
                convolution_factory=convolution_factory,
            )

        def down(in_channels: int, out_channels: int):
            return ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                convolution_factory=curry(convolution)(
                    kernel_size=config.strided_kernel_size, stride=2
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
                    output_padding=0,
                    stride_type="transpose",
                ),
                activation_factory=relu_activation(),
            )

        min_filters = int(config.max_filters / 2 ** config.n_convolutions)

        if config.disable_convolutions:
            first_conv = nn.Identity()
            encoder_decoder = nn.Identity()
            out_conv = nn.Identity()
        else:
            first_conv = nn.Sequential(
                convolution(
                    kernel_size=7, in_channels=channels, out_channels=min_filters,
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
        self._first_conv = first_conv
        self._encoder_decoder = encoder_decoder
        self._out_conv = out_conv
        if config.use_geographic_bias:
            self._input_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
            self._output_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
        else:
            self._input_bias = nn.Identity()
            self._output_bias = nn.Identity()

    def init_hidden(self, input_shape) -> torch.Tensor:
        """
        Initialize the hidden state of the network.

        Args:
            input_shape: shape of the input to the network, should be
                [n_sample, n_tile, n_channels, nx, ny]

        Returns:
            hidden state of the network, of shape
            [n_sample, n_tile, n_channels, nx_coarse, ny_coarse]
        """
        if tuple(input_shape[-4:]) != (6, self.channels, self.nx, self.ny):
            raise ValueError(
                "input_shape must be [n_sample, n_tile, n_channels, nx, ny], "
                "but given shape is {} which does not match init values of {}".format(
                    input_shape, [6, self.channels, self.nx, self.ny]
                )
            )
        hidden_state = torch.zeros(
            size=(
                input_shape[0],
                6,
                self.hidden_channels,
                int(self.nx / (2 ** self.n_convolutions)),
                int(self.ny / (2 ** self.n_convolutions)),
            )
        ).to(DEVICE)
        return hidden_state

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]
            hidden: tensor of shape [batch, tile, channels, x_coarse, y_coarse]

        Returns:
            outputs: tensor of shape [batch, tile, channels, x, y]
            hidden: tensor of shape batch, tile, channels, x_coarse, y_coarse]
        """
        x = self._input_bias(inputs)
        x = self._first_conv(x)
        x, hidden = self._encoder_decoder(x, hidden)
        x = self._out_conv(x)
        outputs: torch.Tensor = self._output_bias(x)
        return outputs, hidden


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

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]
            hidden: tensor of shape [batch, tile, channels_coarse, x_coarse, y_coarse]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        x = self._down(inputs)
        x, hidden = self._lower(x, hidden)
        x = self._up(x)
        return x, hidden

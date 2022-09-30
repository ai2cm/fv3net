import dataclasses
from typing import Tuple
import torch.nn as nn
from toolz import curry
import torch
from ..system import DEVICE
from .modules import (
    ConvBlock,
    ConvolutionFactory,
    FoldFirstDimension,
    single_tile_convolution,
    relu_activation,
    ResnetBlock,
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
    """

    n_convolutions: int = 3
    n_resnet: int = 3
    kernel_size: int = 3
    strided_kernel_size: int = 4
    max_filters: int = 256
    use_geographic_bias: bool = True

    def build(
        self,
        channels: int,
        nx: int,
        ny: int,
        n_time: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            channels: number of input channels
            nx: number of x grid points
            ny: number of y grid points
            n_time: number of timesteps in output timeseries, including input timestep
            convolution: factory for creating all convolutional layers
                used by the network
        """
        return RecurrentGenerator(
            config=self,
            channels=channels,
            nx=nx,
            ny=ny,
            n_time=n_time,
            convolution=convolution,
        ).to(DEVICE)


class GeographicBias(nn.Module):
    """
    Adds a trainable bias vector of shape [6, channels, nx, ny] to the layer input.
    """

    def __init__(self, channels: int, nx: int, ny: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6, channels, nx, ny))

    def forward(self, x):
        return x + self.bias


class RecurrentGenerator(nn.Module):
    def __init__(
        self,
        config: RecurrentGeneratorConfig,
        channels: int,
        nx: int,
        ny: int,
        n_time: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            config: configuration for the network
            channels: number of input and output channels
            nx: number of grid points in x direction
            ny: number of grid points in y direction
            n_time: number of timesteps in output timeseries, including input timestep
            convolution: factory for creating all convolutional layers
                used by the network
        """
        super(RecurrentGenerator, self).__init__()
        self.channels = channels
        self.hidden_channels = config.max_filters
        self.nx = nx
        self.ny = ny
        self.ntime = n_time
        self.n_convolutions = config.n_convolutions

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

        self.first_conv = nn.Sequential(
            convolution(kernel_size=7, in_channels=channels, out_channels=min_filters,),
            FoldFirstDimension(nn.InstanceNorm2d(min_filters)),
            relu_activation()(),
        )
        self.encoder = nn.Sequential(
            *[
                down(
                    in_channels=min_filters * (2 ** i),
                    out_channels=min_filters * (2 ** (i + 1)),
                )
                for i in range(self.n_convolutions)
            ]
        )
        self.resnet = resnet(
            in_channels=config.max_filters, out_channels=config.max_filters
        )

        self.decoder = nn.Sequential(
            *[
                up(
                    in_channels=int(config.max_filters / (2 ** i)),
                    out_channels=int(config.max_filters / (2 ** (i + 1))),
                )
                for i in range(self.n_convolutions)
            ]
        )
        self.out_conv = nn.Sequential(
            convolution(kernel_size=7, in_channels=min_filters, out_channels=channels,),
        )
        if config.use_geographic_bias:
            self.input_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
            self.output_bias = GeographicBias(channels=channels, nx=nx, ny=ny)
        else:
            self.input_bias = nn.Identity()
            self.output_bias = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            outputs: tensor of shape [batch, time, tile, channels, x, y]
        """
        x = self._encode(inputs)
        out_states = [self._decode(x)]
        for _ in range(self.ntime - 1):
            x = self._step(x)
            out_states.append(self._decode(x))
        out = torch.stack(out_states, dim=1)
        return out

    def _encode(self, x: torch.Tensor):
        """
        Transform x from real into latent space.
        """
        x = self.input_bias(x)
        x = self.first_conv(x)
        return self.encoder(x)

    def _step(self, x: torch.Tensor):
        """
        Step latent x forward in time.
        """
        return self.resnet(x)

    def _decode(self, x: torch.Tensor):
        """
        Transform x from latent into real space.
        """
        x = self.decoder(x)
        x = self.out_conv(x)
        return self.output_bias(x)

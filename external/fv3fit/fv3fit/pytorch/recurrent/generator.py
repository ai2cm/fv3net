import dataclasses
from typing import Optional, Tuple
import torch.nn as nn
from toolz import curry
import torch
from ..system import DEVICE
from ..cyclegan.modules import (
    ConvBlock,
    ConvolutionFactory,
    FoldFirstDimension,
    single_tile_convolution,
    relu_activation,
    ResnetBlock,
)
from ..cyclegan.generator import GeographicBias
from torch.utils.checkpoint import checkpoint
import numpy as np
from vcm.grid import get_grid_xyz
from .shapes import RecurrentShape


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
        step_type: type of recurrent step to use, must be one of "resnet" or "conv"
        samples_per_day: number of samples per model day, if given will provide the
            model with two internal features tracking the position of the hour hand on
            a 24 hour clock assuming the starting time of each window is midnight
    """

    n_convolutions: int = 3
    n_resnet: int = 3
    kernel_size: int = 3
    strided_kernel_size: int = 4
    max_filters: int = 256
    use_geographic_bias: bool = True
    use_geographic_features: bool = True
    step_type: str = "resnet"
    samples_per_day: Optional[int] = None

    def __post_init__(self):
        if self.step_type not in ["resnet", "conv"]:
            raise TypeError("step_type must be one of 'resnet' or 'conv'")

    def build(
        self,
        channels: int,
        shape: RecurrentShape,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        """
        Args:
            channels: number of input channels
            shape: shape information for input tensors
            convolution: factory for creating all convolutional layers
                used by the network
        """
        return RecurrentGenerator(
            config=self,
            channels=channels,
            shape=shape,
            convolution=convolution,
            step_type=self.step_type,
        ).to(DEVICE)


class PersistContext(nn.Module):
    def __init__(self, op: nn.Module, context_channels: int):
        super(PersistContext, self).__init__()
        self.op = op
        self.context_channels = context_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        x = self.op(inputs)
        if self.context_channels > 0:
            x = torch.cat(
                tensors=[
                    x,
                    torch.zeros_like(inputs[:, :, -self.context_channels :, :]),
                ],
                dim=-3,
            )
        return x


class SelectChannels(nn.Module):
    """Module which slices the channel dimension."""

    def __init__(self, start: Optional[int], stop: Optional[int], step: Optional[int]):
        super().__init__()
        self._slice = slice(start, stop, step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., channel, x, y)
        """
        return x[..., self._slice, :, :]


class DiurnalFeatures(nn.Module):
    """
    Appends (x, y) features corresponding to the position of the hour hand
    on a 24-hour clock, assuming the initial time is at x, y = 0, 1
    """

    def __init__(self, samples_per_day: int):
        super(DiurnalFeatures, self).__init__()
        self.samples_per_day = samples_per_day
        self.clock = 0

    def reset(self):
        self.clock = 0

    def forward(self, x):
        """
        Args:
            inputs: tensor of shape [sample, channels, x, y]

        Returns:
            outputs: tensor of shape [sample, channels + 2, x, y]
        """
        features = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=DEVICE)
        features[:, 0, :, :] = np.sin(2 * np.pi * self.clock / self.samples_per_day)
        features[:, 1, :, :] = np.cos(2 * np.pi * self.clock / self.samples_per_day)
        self.clock = (self.clock + 1) % self.samples_per_day
        return torch.cat([x, features], dim=-3)


class GeographicFeatures(nn.Module):
    """
    Appends (x, y, z) features corresponding to Eulerian position of each
    gridcell on a unit sphere.
    """

    def __init__(self, nx: int, ny: int):
        super().__init__()
        if nx != ny:
            raise ValueError("this object requires nx=ny")
        self.xyz = torch.as_tensor(
            get_grid_xyz(nx=nx).transpose([0, 3, 1, 2]), device=DEVICE
        ).float()

    def forward(self, x):
        """
        Args:
            x: tensor of shape [sample, tile, channel, x, y]

        Returns:
            tensor of shape [sample, tile, channel, x, y]
        """
        # the fact that this appends instead of prepends is arbitrary but important,
        # this is assumed to be the case elsewhere in the code.
        return torch.concat(
            [x, torch.stack([self.xyz for _ in range(x.shape[0])], dim=0)], dim=-3
        )


class RecurrentGenerator(nn.Module):
    def __init__(
        self,
        config: RecurrentGeneratorConfig,
        channels: int,
        shape: RecurrentShape,
        convolution: ConvolutionFactory = single_tile_convolution,
        step_type: str = "resnet",
    ):
        """
        Args:
            config: configuration for the network
            channels: number of input and output channels
            shape: shape information for input tensors
            convolution: factory for creating all convolutional layers
                used by the network
            step_type: type of recurrent step to use, must be one of "resnet" or "conv"
        """
        super(RecurrentGenerator, self).__init__()
        self.channels = channels
        if config.use_geographic_features:
            xyz_channels = 3
        else:
            xyz_channels = 0
        if config.samples_per_day is not None:
            diurnal_channels = 2
        else:
            diurnal_channels = 0
        self.hidden_channels = config.max_filters
        self.ntime = shape.n_time
        self.n_convolutions = config.n_convolutions

        if step_type == "resnet":

            def step(in_channels: int, out_channels: int, context_channels: int = 0):
                if in_channels != out_channels:
                    raise ValueError(
                        "resnet must have same number of output channels as "
                        "input channels, since the inputs are added to the outputs"
                    )
                resnet_blocks = [
                    ResnetBlock(
                        channels=in_channels,
                        context_channels=context_channels,
                        convolution_factory=curry(convolution)(
                            kernel_size=config.kernel_size
                        ),
                        activation_factory=relu_activation(),
                    )
                    for _ in range(config.n_resnet)
                ]
                return nn.Sequential(*resnet_blocks)

        elif step_type == "conv":

            def step(in_channels: int, out_channels: int, context_channels: int = 0):
                conv_block = nn.Sequential(
                    ConvBlock(
                        in_channels=in_channels + context_channels,
                        out_channels=out_channels,
                        convolution_factory=curry(convolution)(
                            kernel_size=config.kernel_size
                        ),
                        activation_factory=relu_activation(),
                    ),
                    ConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        convolution_factory=curry(convolution)(
                            kernel_size=config.kernel_size
                        ),
                        activation_factory=relu_activation(),
                    ),
                )
                return PersistContext(conv_block, context_channels=context_channels)

        else:
            raise ValueError(
                f"Unknown step type {step_type}, expected 'resnet' or 'conv'"
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

        self.first_conv = nn.Sequential(
            convolution(
                kernel_size=7,
                in_channels=channels + xyz_channels,
                out_channels=min_filters,
            ),
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
        context_modules = []
        if config.use_geographic_features:
            nx_resnet = shape.nx // int(2 ** config.n_convolutions)
            context_modules.append(GeographicFeatures(nx=nx_resnet, ny=nx_resnet))
        if config.samples_per_day is not None:
            self.clock: Optional[DiurnalFeatures] = DiurnalFeatures(
                samples_per_day=config.samples_per_day
            )
            context_modules.append(FoldFirstDimension(self.clock))
        else:
            self.clock = None

        self.resnet = nn.Sequential(
            *context_modules,
            step(
                in_channels=config.max_filters,
                out_channels=config.max_filters,
                context_channels=xyz_channels + diurnal_channels,
            ),
            SelectChannels(0, config.max_filters, 1),
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
            self.input_bias = GeographicBias(
                channels=channels, nx=shape.nx, ny=shape.ny
            )
            self.output_bias = GeographicBias(
                channels=channels, nx=shape.nx, ny=shape.ny
            )
        else:
            self.input_bias = nn.Identity()
            self.output_bias = nn.Identity()
        if config.use_geographic_features:
            self.input_bias = nn.Sequential(
                self.input_bias, GeographicFeatures(nx=shape.nx, ny=shape.ny)
            )

    def forward(
        self, inputs: torch.Tensor, ntime: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            outputs: tensor of shape [batch, time, tile, channels, x, y]
        """
        if hasattr(self, "clock") and self.clock is not None:
            self.clock.reset()
        if ntime is None:
            ntime = self.ntime
        x = self._encode(inputs)
        out_states = [self._decode(x)]
        for _ in range(ntime - 1):
            x = checkpoint(self._step, x)
            out_states.append(self._decode(x))
        out = torch.stack(out_states, dim=1)
        return out

    def _encode(self, inputs: torch.Tensor):
        """
        Transform x from real into latent space.
        """
        x = self.input_bias(inputs)
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

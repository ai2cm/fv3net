import logging

from typing import Callable, Literal, Optional, Protocol, Tuple
import torch.nn as nn
import torch
from vcm.grid import get_grid_xyz, get_grid
from fv3fit.pytorch.system import DEVICE
import numpy as np

logger = logging.getLogger(__name__)

HOURS_PER_DEG_LONGITUDE = 1.0 / 15
SECONDS_PER_DAY = 24 * 60 * 60


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


class DiscardTime(nn.Module):
    """
    Takes a tuple of (time, state) and returns state.

    Useful as an alternative for GeographicFeatures which does nothing.
    """

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return x[1]


class GeographicFeatures(nn.Module):
    """
    Appends (time_x, time_y, x, y, z) features corresponding to the local position
    of the hour hand on a 24h clock, and
    the Eulerian position of each gridcell on a unit sphere.
    """

    N_FEATURES = 5

    def __init__(self, nx: int, ny: int):
        super().__init__()
        if nx != ny:
            raise ValueError("this object requires nx=ny")
        self.xyz = torch.as_tensor(
            # transpose to move channel dimension (last) to second dim
            get_grid_xyz(nx=nx).transpose([0, 3, 1, 2]),
            device=DEVICE,
        ).float()
        lon, lat = get_grid(nx=nx)
        lon = lon[:, None, :, :]  # insert channel dimension
        lat = lat[:, None, :, :]  # insert channel dimension
        self.local_time_zero_radians = torch.as_tensor(lon, device=DEVICE).float()
        self.cos_lat = torch.as_tensor(np.cos(lat), device=DEVICE).float()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: A tuple containing a tensor of shape (batch, window)
                with the time as
                a number of seconds since any time corresponding to midnight at
                longitude 0,
                and a tensor of shape (batch, window, tile, in_channels, x, y)

        Returns:
            tensor of shape [batch, window, tile, channel, x, y]
        """
        # TODO: this is a hack to support the previous API before time was added,
        # remove this try-except once the API is stable
        try:
            time, x = inputs
            assert len(time.shape) == 1, "time must be a 1D tensor"
            local_time_offset_radians = (
                time % SECONDS_PER_DAY / SECONDS_PER_DAY * 2 * np.pi
            )
            local_time = (
                self.local_time_zero_radians[None, :]
                + local_time_offset_radians[:, None, None, None, None]
            )
            if hasattr(self, "cos_lat"):
                time_x = torch.sin(local_time) * self.cos_lat[None, :]
                time_y = torch.cos(local_time) * self.cos_lat[None, :]
            else:
                time_x = torch.sin(local_time)
                time_y = torch.cos(local_time)
            xyz = torch.stack([self.xyz for _ in range(x.shape[0])])
            geo_features = torch.cat([time_x, time_y, xyz], dim=2)
        except ValueError:
            x = inputs
            geo_features = torch.stack([self.xyz for _ in range(x.shape[0])])

        # the fact that this appends instead of prepends is arbitrary but important,
        # this is assumed to be the case elsewhere in the code.
        return torch.cat([x, geo_features], dim=-3)


class GeographicBias(nn.Module):
    """
    Adds a trainable bias vector of shape [6, channels, nx, ny] to the layer input.
    """

    def __init__(self, channels: int, nx: int, ny: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6, channels, nx, ny))

    def forward(self, x):
        return x + self.bias


class ConvolutionFactory(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        output_padding: int = 0,
        stride: int = 1,
        stride_type: Literal["regular", "transpose"] = "regular",
        bias: bool = True,
    ) -> nn.Module:
        """
        Create a convolutional layer.

        Layer takes in and returns tensors of shape [batch, tile, channels, x, y].

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of the convolution kernel
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


class FoldFirstDimension(nn.Module):
    """
    Module wrapping a module which takes e.g. [batch, channel, x, y] data into one
    which takes [batch, tile, channel, x, y] data by folding the first dimension
    into the second dimension.
    """

    def __init__(self, wrapped):
        super(FoldFirstDimension, self).__init__()
        self._wrapped = wrapped

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(-1, *inputs.shape[2:])
        x = self._wrapped(x)
        return x.reshape(inputs.shape[0], inputs.shape[1], *x.shape[1:])


def single_tile_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_padding: int = 0,
    stride: int = 1,
    stride_type: Literal["regular", "transpose"] = "regular",
    padding: Optional[int] = None,
    bias: bool = True,
) -> ConvolutionFactory:
    """
    Construct a convolutional layer for single tile data (like images).

    Layer takes in and returns tensors of shape [batch, tile, channels, x, y].

    Args:
        kernel_size: size of the convolution kernel
        padding: padding to apply to the input, should be an integer or "same"
        output_padding: argument used for transpose convolution
        stride: stride of the convolution
        stride_type: type of stride, one of "regular" or "transpose"
        bias: whether to include a bias vector in the produced layers
    """
    if padding is None:
        padding = int((kernel_size - 1) // 2)
    if stride == 1:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
    elif stride_type == "regular":
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif stride_type == "transpose":
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
    else:
        raise ValueError(f"Invalid stride_type: {stride_type}")
    return FoldFirstDimension(conv)


def halo_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_padding: int = 0,
    stride: int = 1,
    stride_type: Literal["regular", "transpose"] = "regular",
    padding: Optional[int] = None,
    bias: bool = True,
) -> ConvolutionFactory:
    """
    Construct a convolutional layer that pads with halo data before applying conv2d.

    Padding is such that the output shape should be the input shape divided by the
    stride for regular convolution. For transpose convolution, the output shape
    will be the input shape multiplied by the stride.

    Layer takes in and returns tensors of shape [batch, tile, channels, x, y].

    There may be reason to prefer using an even kernel size for strided convolutions,
    and an odd kernel size for non-strided convolutions. For the non-strided
    case, an even kernel size gives a symmetric kernel as desired. We've only reasoned
    this for a stride of 2, but it should hold for any even stride. However, this
    should be tested in practice. We've found an odd kernel size can give better model
    performance.

    The strided case (which uses a stride of 2) is more complex, but an even
    kernel size gives a symmetric kernel. This is because a (2, 2) patch of the input
    domain corresponds to a (1, 1) patch of the output domain. We would need to have a
    (2, 2) patch in the center of the kernel (which is in input space) in order for
    the kernel to be symmetric about the (1, 1) center in output space. This means the
    kernel must be even-sized in input space.

    For example, consider a 6x6 kernel. In the ASCII diagram below, an X corresponds
    to a gridcell in the input space, and gridcells in the output space are separated
    by lines:

        X X | X X | X X
        X X | X X | X X
        ---------------
        X X | X X | X X
        X X | X X | X X
        ---------------
        X X | X X | X X
        X X | X X | X X

    The kernel is symmetric about the center gridcell in the output space, as desired.

    The transpose strided case (where the input domain is higher-resolution than the
    output domain, by a factor of 2) is even more complicated.

    Consider a 4x4 kernel. In the ASCII diagram below, the full diagram corresponds to
    one example output gridcell. The X values correspond to valid data in the input
    domain, while the O values correspond to zero-padding injected by the transpose
    stride.

        X O X O
        O O O O
        X O X O
        O O O O

    Another example might look like:

        O O O O
        O X O X
        O O O O
        O X O X

    The even kernel size is desirable in this case for a different reason. Not because
    of a symmetric kernel, but because an even kernel size means all output values
    are "connected" to the same number of input values. If the kernel size were odd,
    say 3x3 in this case, the first example would be connected to 4 values in the input
    space while the second example would be connected to only one:

        X O X
        O O O
        X O X

        O O O
        O X O
        O O O

    This can also be represented with the diagram above, where we stride the input
    domain (lower resolution) values into the output (convolution) domain:

        X O | X O | X O
        O O | O O | O O
        ---------------
        X O | S O | X O
        O O | O O | O O
        ---------------
        X O | X O | X O
        O O | O O | O O

    Where S is still a valid-input value, but is marked specially as the first point
    in the compute domain, as described below.

    Let's walk through what happens with a 4x4 kernel, specifically for the
    "halo convolution" case. What we want is that each output point depends on the
    same number of input points, to have directional invariance.
    In this case we specifically want each output point to be connected to the
    4 closest input points, e.g. the top-left output within a larger gridcell depends
    on the value in that gridcell, and the values above and to the left of it.

    In this case we append 2 halo points to the input, so you can consider the
    left and top two rows to be in the computational halo (don't correspond to
    output locations).
    The first output gridcell for the convolution will be connected only to the
    top-leftmost halo point, via its bottom-rightmost kernel cell. However,
    we crop the output of the convolution - specifically for a 4x4 kernel we crop
    2 edge points for the padded halo, and 1 more for the kernel "extension effect".
    This means the first output gridcell S reads from exactly the 4x4 segment of the
    top left of the domain above, which contains exactly the 4 points desired.

    The second output just to the right of S will read from the two X points to the
    right instead of the two points to the left, and the output just below S will
    depend on the two points below instead of the two points above. So we have the
    desired directional invariance.

    If you would like to see more visualizations of transpose convolution,
    we suggest this github repo: https://github.com/vdumoulin/conv_arithmetic

    Args:
        in_channels: number of input channels,
        out_channels: number of output channels,
        kernel_size: size of the convolution kernel
        output_padding: argument used for transpose convolution
        stride: stride of the convolution
        stride_type: type of stride, one of "regular" or "transpose"
        padding: if given, override automatic padding calculation and use this
            number of halo points instead
        bias: whether to include a bias vector in the produced layers
    """
    if padding is None:
        padding = int((kernel_size - 1) // 2)
    append = AppendHalos(n_halo=padding)
    conv = single_tile_convolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        output_padding=output_padding,
        padding=0,  # we already padded with AppendHalos
        stride=stride,
        stride_type=stride_type,
        bias=bias,
    )
    if stride_type == "transpose":
        # have to crop halo points from the output, as pytorch has no option to
        # only output a subset of the domain for ConvTranspose2d
        #
        # padding * stride is the part of the output domain corresponding
        # to the upscaled halo points
        #
        # transpose convolution adds (kernel_size - 1) / 2 more points which
        # correspond to every point in the upsampled domain where the kernel
        # can read from at least one point in the input domain
        #
        # we remove both of these to keep only the "compute domain"
        # TODO: may be able to replace this Crop with the `padding` argument
        # to ConvTranspose2d as it behaves opposite to the padding argument
        # for Conv2d
        conv = nn.Sequential(
            conv, Crop(n_halo=padding * stride + int(kernel_size - 1) // 2)
        )
    return nn.Sequential(append, conv)


def convolution_factory_from_name(name: str) -> ConvolutionFactory:
    if name == "conv2d":
        return single_tile_convolution
    elif name == "halo_conv2d":
        return halo_convolution
    else:
        raise ValueError(f"Unknown convolution type: {name}")


class Crop(nn.Module):
    def __init__(self, n_halo):
        super(Crop, self).__init__()
        self.n_halo = n_halo

    def forward(self, x):
        return x[
            ..., self.n_halo : -self.n_halo, self.n_halo : -self.n_halo
        ].contiguous()


class AppendHalos(nn.Module):

    """
    Module which appends horizontal halos to the input tensor.

    Args:
        n_halo: size of the halo to append
    """

    def __init__(self, n_halo: int):
        super(AppendHalos, self).__init__()
        self.n_halo = n_halo

    def extra_repr(self) -> str:
        return super().extra_repr() + f"n_halo={self.n_halo}"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channel, x, y]
        """
        corner = torch.zeros_like(inputs[:, 0, :, : self.n_halo, : self.n_halo])
        if self.n_halo > 0:
            with_halo = []
            for _ in range(6):
                tile = []
                for _ in range(3):
                    column = []
                    for _ in range(3):
                        column.append([None, None, None])  # row
                    tile.append(column)
                with_halo.append(tile)

            for i_tile in range(6):
                with_halo[i_tile][1][1] = inputs[:, i_tile, :, :, :]
                with_halo[i_tile][0][0] = corner
                with_halo[i_tile][0][2] = corner
                with_halo[i_tile][2][0] = corner
                with_halo[i_tile][2][2] = corner
                # The comments below about what to write where were verified by
                # printing out and gluing together a 3D cube of the tile faces with
                # their local x-y axes indicated, and looking at the boundaries of
                # the 3D shape. The layout used is the "staircase" logical alignment
                # described in Chen (2020).
                # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002280
                if i_tile % 2 == 0:  # even tile
                    # y-start edge
                    # we must make data contiguous after rotating 90 degrees because
                    # the MPS backend doesn't properly manage strides when concatenating
                    # arrays
                    with_halo[i_tile][0][1] = torch.rot90(
                        inputs[
                            :, (i_tile - 2) % 6, :, :, -self.n_halo :
                        ],  # write tile 4 to tile 0
                        k=-1,  # relative rotation of tile 0 with respect to tile 4
                        dims=(2, 3),
                    ).contiguous()
                    # x-start edge
                    with_halo[i_tile][1][0] = inputs[
                        :, (i_tile - 1) % 6, :, :, -self.n_halo :
                    ]  # write tile 5 to tile 0
                    # x-end edge
                    with_halo[i_tile][2][1] = inputs[
                        :,
                        (i_tile + 1) % 6,
                        :,
                        : self.n_halo,
                        :,  # write tile 5 to tile 0
                    ]
                    # y-end edge
                    with_halo[i_tile][1][2] = torch.rot90(
                        inputs[
                            :, (i_tile + 2) % 6, :, : self.n_halo, :
                        ],  # write tile 2 to tile 0
                        k=1,  # relative rotation of tile 0 with respect to tile 2
                        dims=(2, 3),
                    ).contiguous()
                else:  # odd tile
                    # y-start edge
                    with_halo[i_tile][0][1] = inputs[
                        :,
                        (i_tile - 1) % 6,
                        :,
                        -self.n_halo :,
                        :,  # write tile 0 to tile 1
                    ]
                    # x-start edge
                    with_halo[i_tile][1][0] = torch.rot90(
                        inputs[
                            :, (i_tile - 2) % 6, :, -self.n_halo :, :
                        ],  # write tile 5 to tile 1
                        k=1,  # relative rotation of tile 1 with respect to tile 5
                        dims=(2, 3),
                    ).contiguous()
                    # x-end edge
                    with_halo[i_tile][2][1] = torch.rot90(
                        inputs[
                            :, (i_tile + 2) % 6, :, :, : self.n_halo
                        ],  # write tile 3 to tile 1
                        k=-1,  # relative rotation of tile 1 with respect to tile 3
                        dims=(2, 3),
                    ).contiguous()
                    # y-end edge
                    with_halo[i_tile][1][2] = inputs[
                        :, (i_tile + 1) % 6, :, :, : self.n_halo
                    ]  # write tile 2 to tile 1

            for i_tile in range(6):
                for i_col in range(3):
                    with_halo[i_tile][i_col] = torch.cat(
                        tensors=with_halo[i_tile][i_col], dim=-1
                    )
                with_halo[i_tile] = torch.cat(tensors=with_halo[i_tile], dim=-2)
            with_halo = torch.stack(tensors=with_halo, dim=1)

        else:
            with_halo = inputs

        return with_halo


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
        context_channels: int = 0,
    ):
        """
        Args:
            channels: number of filters in the internal
                convolutional layers, input and output channels for the block
                are channels + context_channels
            convolution_factory: factory for creating convolutional layers
            activation_factory: factory for creating activation layers
            context_channels: if given, this number of channels at the end of the input
                are treated as context channels and passed through to the output
                of the resnet block, but not modified.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=channels + context_channels,
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
        self.context_channels = context_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        g = self.conv_block(inputs)
        if self.context_channels > 0:
            g = torch.cat(
                tensors=[
                    g,
                    torch.zeros_like(inputs[:, :, -self.context_channels :, :]),
                ],
                dim=-3,
            )
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
            FoldFirstDimension(nn.InstanceNorm2d(out_channels)),
            activation_factory(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        return self.conv_block(inputs)

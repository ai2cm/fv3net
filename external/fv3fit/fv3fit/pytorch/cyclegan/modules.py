import logging
import functools

from typing import Callable, Literal, Protocol
import torch.nn as nn
import torch

logger = logging.getLogger(__name__)


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


class FoldTileDimension(nn.Module):
    """
    Module wrapping a module which takes [batch, channel, x, y] data into one
    which takes [batch, tile, channel, x, y] data by folding the tile dimension
    into the batch dimension.
    """

    def __init__(self, wrapped):
        super(FoldTileDimension, self).__init__()
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
    padding = "same"
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
            padding=(padding, padding),
            output_padding=output_padding,
            bias=bias,
        )
    else:
        raise ValueError(f"Invalid stride_type: {stride_type}")
    return FoldTileDimension(conv)


def halo_convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_padding: int = 0,
    stride: int = 1,
    stride_type: Literal["regular", "transpose"] = "regular",
    bias: bool = True,
) -> ConvolutionFactory:
    """
    Construct a convolutional layer that pads with halo data before applying conv2d.

    Padding is such that the output shape should be the input shape divided by the
    stride for regular convolution. For transpose convolution, the output shape
    will be the input shape multiplied by the stride.

    Layer takes in and returns tensors of shape [batch, tile, channels, x, y].

    Args:
        kernel_size: size of the convolution kernel
        output_padding: argument used for transpose convolution
        stride: stride of the convolution
        stride_type: type of stride, one of "regular" or "transpose"
        bias: whether to include a bias vector in the produced layers
    """
    if stride_type == "transpose":
        padding = int((kernel_size - 1) // 2 * stride)
    else:
        padding = int((kernel_size - 1) // 2)
    append = AppendHalos(n_halo=padding)
    conv = single_tile_convolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        output_padding=output_padding,
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
        conv = nn.Sequential(
            conv, Crop(n_halo=padding * stride + int(kernel_size - 1) // 2)
        )
    return nn.Sequential(append, conv)


def cpu_only(method):
    """
    Decorator to mark a method as only being supported on the CPU.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        original_device = args[0].device
        args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        return method(self, *args, **kwargs).to(original_device)

    return wrapper


class Crop(nn.Module):
    def __init__(self, n_halo):
        super(Crop, self).__init__()
        self.n_halo = n_halo

    @cpu_only
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

    @cpu_only
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
                # we must make data contiguous after rotating 90 degrees because
                # the MPS backend doesn't properly manage strides when concatenating
                # arrays
                if i_tile % 2 == 0:  # even tile
                    # south edge
                    with_halo[i_tile][0][1] = torch.rot90(
                        inputs[
                            :, (i_tile - 2) % 6, :, :, -self.n_halo :
                        ],  # write tile 4 to tile 0
                        k=-1,  # relative rotation of tile 0 with respect to tile 5
                        dims=(2, 3),
                    ).contiguous()
                    # west edge
                    with_halo[i_tile][1][0] = inputs[
                        :, (i_tile - 1) % 6, :, :, -self.n_halo :
                    ]  # write tile 5 to tile 0
                    # east edge
                    with_halo[i_tile][2][1] = inputs[
                        :,
                        (i_tile + 1) % 6,
                        :,
                        : self.n_halo,
                        :,  # write tile 1 to tile 0
                    ]
                    # north edge
                    with_halo[i_tile][1][2] = torch.rot90(
                        inputs[
                            :, (i_tile + 2) % 6, :, : self.n_halo, :
                        ],  # write tile 2 to tile 0
                        k=1,  # relative rotation of tile 0 with respect to tile 2
                        dims=(2, 3),
                    ).contiguous()
                else:  # odd tile
                    # south edge
                    with_halo[i_tile][0][1] = inputs[
                        :,
                        (i_tile - 1) % 6,
                        :,
                        -self.n_halo :,
                        :,  # write tile 0 to tile 1
                    ]
                    # west edge
                    with_halo[i_tile][1][0] = torch.rot90(
                        inputs[
                            :, (i_tile - 2) % 6, :, -self.n_halo :, :
                        ],  # write tile 5 to tile 1
                        k=1,  # relative rotation of tile 1 with respect to tile 5
                        dims=(2, 3),
                    ).contiguous()
                    # east edge
                    with_halo[i_tile][2][1] = torch.rot90(
                        inputs[
                            :, (i_tile + 2) % 6, :, :, : self.n_halo
                        ],  # write tile 3 to tile 1
                        k=-1,  # relative rotation of tile 1 with respect to tile 3
                        dims=(2, 3),
                    ).contiguous()
                    # north edge
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
    ):
        """
        Args:
            channels: number of input channels and filters in the convolutional layers
            convolution_factory: factory for creating convolutional layers
            activation_factory: factory for creating activation layers
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, tile, channels, x, y]

        Returns:
            tensor of shape [batch, tile, channels, x, y]
        """
        g = self.conv_block(inputs)
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
            FoldTileDimension(nn.InstanceNorm2d(out_channels)),
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

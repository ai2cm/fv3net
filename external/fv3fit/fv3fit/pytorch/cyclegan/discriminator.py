import dataclasses
import torch.nn as nn
from toolz import curry
import torch
from .modules import (
    ConvolutionFactory,
    GeographicFeatures,
    GeographicBias,
    DiscardTime,
    single_tile_convolution,
    leakyrelu_activation,
    ConvBlock,
)


@dataclasses.dataclass
class DiscriminatorConfig:
    """
    Configuration for a discriminator network.

    Follows the architecture of Zhu et al. 2017, https://arxiv.org/abs/1703.10593.
    Uses a series of strided convolutions with leaky ReLU activations, followed
    by two convolutional layers.

    Args:
        n_convolutions: number of strided convolutional layers before the
            final convolutional output layer, must be at least 1
        kernel_size: size of convolutional kernels
        strided_kernel_size: size of convolutional kernels in the
            strided convolutions
        max_filters: maximum number of filters in any convolutional layer,
            equal to the number of filters in the final strided convolutional layer
        use_geographic_features: if True, include a layer that appends
            geographic features to the input data.
        use_geographic_embedded_bias: if True, include a layer that adds a trainable
            bias vector after the initial encoding layer. This bias is a
            function of horizontal coordinates.
    """

    n_convolutions: int = 3
    kernel_size: int = 3
    strided_kernel_size: int = 3
    max_filters: int = 256
    use_geographic_features: bool = True
    use_geographic_embedded_bias: bool = False

    def build(
        self,
        channels: int,
        nx: int,
        ny: int,
        convolution: ConvolutionFactory = single_tile_convolution,
    ):
        return Discriminator(
            in_channels=channels,
            n_convolutions=self.n_convolutions,
            kernel_size=self.kernel_size,
            strided_kernel_size=self.strided_kernel_size,
            max_filters=self.max_filters,
            nx=nx,
            ny=ny,
            convolution=convolution,
            use_geographic_features=self.use_geographic_features,
            use_geographic_embedded_bias=self.use_geographic_embedded_bias,
        )


class Discriminator(nn.Module):

    # analogous to NLayerDiscriminator at
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    def __init__(
        self,
        in_channels: int,
        n_convolutions: int,
        kernel_size: int,
        strided_kernel_size: int,
        max_filters: int,
        nx: int,
        ny: int,
        convolution: ConvolutionFactory = single_tile_convolution,
        use_geographic_features: bool = True,
        use_geographic_embedded_bias: bool = False,
    ):
        """
        Args:
            in_channels: number of input channels
            n_convolutions: number of strided convolutional layers before the
                final convolutional output layers, must be at least 1
            kernel_size: size of non-strided convolutional kernels
            strided_kernel_size: size of 2-strided convolutional kernels
            max_filters: maximum number of filters in any convolutional layer,
                equal to the number of filters in the final strided convolutional layer
                and in the convolutional layer just before the output layer
            nx: number of grid points in the x direction
            ny: number of grid points in the y direction
            convolution: factory for creating all convolutional layers
            use_geographic_features: if True, include a layer that appends
                geographic features to the input data.
            use_geographic_embedded_bias: if True, include a layer that adds a
                trainable bias vector after the initial encoding layer that is
                a function of horizontal coordinates.
        """
        super(Discriminator, self).__init__()
        if n_convolutions < 1:
            raise ValueError("n_convolutions must be at least 1")
        # max_filters = min_filters * 2 ** (n_convolutions - 1), therefore
        min_filters = int(max_filters / 2 ** (n_convolutions - 1))
        # first convolutional block must not have instance normalization, so that the
        # discriminator can use information about the mean and standard deviation of
        # the input data (generated images)
        if use_geographic_features:
            self._geographic_features = GeographicFeatures(nx=nx, ny=ny)
            in_channels += GeographicFeatures.N_FEATURES
        else:
            self._geographic_features = DiscardTime()

        convs = []
        convs.append(
            convolution(
                in_channels=in_channels,
                out_channels=min_filters,
                kernel_size=strided_kernel_size,
                stride=2,
            )
        )
        if use_geographic_embedded_bias:
            convs.append(
                GeographicBias(channels=min_filters, nx=int(nx / 2), ny=int(ny / 2))
            )
        convs.append(leakyrelu_activation(negative_slope=0.2, inplace=True)())
        # we've already defined the first strided convolutional layer, so start at 1
        for i in range(1, n_convolutions):
            convs.append(
                ConvBlock(
                    in_channels=min_filters * 2 ** (i - 1),
                    out_channels=min_filters * 2 ** i,
                    convolution_factory=curry(convolution)(
                        kernel_size=strided_kernel_size, stride=2
                    ),
                    activation_factory=leakyrelu_activation(
                        negative_slope=0.2, inplace=True
                    ),
                )
            )
        # final_conv isn't strided so it's not included in the n_convolutions count
        final_conv = ConvBlock(
            in_channels=max_filters,
            out_channels=max_filters,
            convolution_factory=curry(convolution)(kernel_size=kernel_size),
            activation_factory=leakyrelu_activation(negative_slope=0.2, inplace=True),
        )
        patch_output = convolution(
            kernel_size=3, in_channels=max_filters, out_channels=1
        )
        self._sequential = nn.Sequential(*convs, final_conv, patch_output)

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: a tensor of shape (batch, 1) with the time as seconds since 1970-01-01
            state: a tensor of shape (batch, tile, in_channels, height, width)

        Returns:
            tensor of shape (batch, tile, 1, height, width)
        """
        x = self._geographic_features((time, state))
        return self._sequential(x)

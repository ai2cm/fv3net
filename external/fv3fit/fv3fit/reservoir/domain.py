import fsspec
import numpy as np
import tensorflow as tf
from typing import Sequence
import yaml
import pace.util


def slice_along_axis(arr: np.ndarray, inds: slice, axis: int = 0):
    # https://stackoverflow.com/a/37729566
    # For slicing ndarray along a dynamically specified axis
    # same as np.take() but does not make a copy of the data
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


class RankDivider:
    def __init__(
        self,
        subdomain_layout: Sequence[int],
        rank_dims: Sequence[str],
        rank_extent: Sequence[int],  # shape of full data, including overlap
        overlap: int,
    ):
        """ Divides a rank of data into subdomains for use in training.
        Args:
            subdomain_layout: tuple describing subdomain grid within the rank
            rank_dims: order of dimensions in data. If using time series data, 'time'
                must be the first dimension.
            rank_extent: Shape of full data. This includes any halo cells from
                overlap into neighboring ranks.
            overlap: number of cells surrounding each subdomain to include when
                taking subdomain data.

        Ex. I want to train reservoirs on 4x4 subdomains with 4 cells of overlap
        across subdomains. The data is preprocessed and saved as 1 C48 tile per rank,
        with n_halo=4. I would initialize the RankDivider as
            RankDivider(
                subdomain_layout=[12, 12],
                rank_dims=["time", "x", "y", "z"],
                rank_extent=[n_timesteps, 56, 56, 79],
                overlap=4,
            )
        """
        self.subdomain_layout = subdomain_layout
        if "time" in rank_dims:
            if rank_dims[0] != "time":
                raise ValueError("'time' must be first dimension.")
        if not {"x", "y"}.issubset(rank_dims):
            raise ValueError(
                "'x' and 'y' dims must be in the rank_dims of the RankDivider"
            )
        self.rank_dims = rank_dims
        self.overlap = overlap
        self.rank_extent = rank_extent
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]

        self._x_ind = rank_dims.index("x")
        self._y_ind = rank_dims.index("y")

        self._partitioner = pace.util.TilePartitioner(subdomain_layout)

        # dimensions of rank data without the halo points. Useful for slice calculation.
        self._rank_extent_without_overlap = self._get_rank_extent_without_overlap(
            rank_extent, overlap
        )

    def dump(self, path):
        metadata = {
            "subdomain_layout": list(self.subdomain_layout),
            "rank_dims": self.rank_dims,
            "overlap": self.overlap,
            "rank_extent": self.rank_extent,
        }
        with fsspec.open(path, "w") as f:
            f.write(yaml.dump(metadata))

    @classmethod
    def load(cls, path):
        with fsspec.open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def get_subdomain_extent(self, with_overlap: bool):
        subdomain_xy_size = (
            self._rank_extent_without_overlap[self._x_ind] // self.subdomain_layout[0]
        )
        if with_overlap:
            subdomain_xy_size += 2 * self.overlap

        subdomain_extent = list(self.rank_extent)
        subdomain_extent[self._x_ind] = subdomain_xy_size
        subdomain_extent[self._y_ind] = subdomain_xy_size
        return tuple(subdomain_extent)

    def subdomain_slice(self, subdomain_index: int, with_overlap: bool):
        # first get the slice indices w/o overlap points for XY data without halo,
        # then calculate adjustments when the overlap cells are included
        slice_ = list(
            self._partitioner.subtile_slice(
                rank=subdomain_index,
                global_dims=self.rank_dims,
                global_extent=self._rank_extent_without_overlap,
            )
        )
        x_slice_ = slice_[self._x_ind]
        y_slice_ = slice_[self._y_ind]

        if with_overlap:
            x_slice_updated = slice(
                x_slice_.start, x_slice_.stop + 2 * self.overlap, None
            )
            y_slice_updated = slice(
                y_slice_.start, y_slice_.stop + 2 * self.overlap, None
            )

        else:
            # The data includes the overlap on the sides of the full rank, so exclude
            # the rank halo region if retrieving the subdomains without overlap cells.
            x_slice_updated = slice(
                x_slice_.start + self.overlap, x_slice_.stop + self.overlap, None
            )
            y_slice_updated = slice(
                y_slice_.start + self.overlap, y_slice_.stop + self.overlap, None
            )

        slice_[self._x_ind] = x_slice_updated
        slice_[self._y_ind] = y_slice_updated
        return tuple(slice_)

    def _get_rank_extent_without_overlap(
        self, data_shape: Sequence[int], overlap: int
    ) -> Sequence[int]:
        extent_without_halos = list(data_shape)
        extent_without_halos[self._x_ind] = (
            extent_without_halos[self._x_ind] - 2 * overlap
        )
        extent_without_halos[self._y_ind] = (
            extent_without_halos[self._y_ind] - 2 * overlap
        )
        return tuple(extent_without_halos)

    def get_subdomain_tensor_slice(
        self, tensor_data: tf.Tensor, subdomain_index: int, with_overlap: bool
    ) -> tf.Tensor:

        subdomain_slice = self.subdomain_slice(subdomain_index, with_overlap)
        tensor_data_xsliced = slice_along_axis(
            arr=tensor_data, inds=subdomain_slice[self._x_ind], axis=self._x_ind
        )
        tensor_data_xy_sliced = slice_along_axis(
            arr=tensor_data_xsliced, inds=subdomain_slice[self._y_ind], axis=self._y_ind
        )
        return tensor_data_xy_sliced

    def unstack_subdomain(self, tensor, with_overlap: bool):
        # Takes a flattened subdomain and reshapes it back into its original
        # x and y dims
        unstacked_shape = self.get_subdomain_extent(with_overlap=with_overlap)[1:]
        expected_stacked_size = np.prod(unstacked_shape)
        if tensor.shape[-1] != expected_stacked_size:
            raise ValueError(
                "Dimension of each stacked sample expected to be "
                f"{expected_stacked_size} (product of {unstacked_shape})."
            )
        unstacked_shape = (tensor.shape[0], *unstacked_shape)
        return np.reshape(tensor, unstacked_shape)


def stack_time_series_samples(tensor):
    # Used to reshape a subdomains into a flat columns.
    # Assumes time is the first dimension
    n_samples = tensor.shape[0]
    return np.reshape(tensor, (n_samples, -1))

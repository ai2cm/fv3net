import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple
import xarray as xr

import pace.util
from fv3fit.keras._models.shared.halos import append_halos

Layout = Tuple[int, int]


def slice_along_axis(arr: np.ndarray, inds: slice, axis: int = 0):
    # https://stackoverflow.com/a/37729566
    # For slicing ndarray along a dynamically specified axis
    # same as np.take() but does not make a copy of the data
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


class CubedsphereDivider:
    def __init__(
        self,
        tile_layout: Layout,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
    ):
        """ Class for dividing global cubedsphere data into ranks for preprocessing.

        Args:
            tile_layout: tuple describing ranks in a tile
            global_dims:  order of dimensions in data.
            global_extent: size of dimensions in the order of global_dims
        """
        self.tile_layout = tile_layout
        if not {"x", "y", "tile"} <= set(global_dims):
            raise ValueError("Data to be divided must have dims 'x', 'y', and 'tile'.")
        self.global_dims = global_dims
        self.global_extent = global_extent
        tile_partitioner = pace.util.TilePartitioner(layout=tile_layout)
        self.cubedsphere_partitioner = pace.util.CubedSpherePartitioner(
            tile=tile_partitioner
        )
        self.total_ranks = self.cubedsphere_partitioner.total_ranks

    def get_rank_data(self, dataset: xr.Dataset, rank: int, overlap: int) -> xr.Dataset:
        """ Returns dataset of a single rank of data, including overlap cells on edges

        Args:
            dataset: global dataset to divide
            rank: index of rank to return
            overlap: number of edge cells to include around rank
        """

        subtile_slice = self.cubedsphere_partitioner.subtile_slice(
            rank,
            global_dims=self.global_dims,
            global_extent=self.global_extent,
            overlap=False,
        )
        subtile_selection = {
            dim: selection
            for (dim, selection) in zip(self.global_dims, subtile_slice)
            if dim in dataset.dims
        }
        metadata = {
            "overlap": overlap,
            "x_start_without_overlap": subtile_selection["x"].start,
            "x_stop_without_overlap": subtile_selection["x"].stop,
            "y_start_without_overlap": subtile_selection["y"].start,
            "y_stop_without_overlap": subtile_selection["y"].stop,
            "tile": subtile_selection["tile"],
        }
        if overlap > 0:
            dataset_ = append_halos(dataset, overlap)
            x_slice = subtile_selection["x"]
            subtile_selection["x"] = slice(
                x_slice.start, x_slice.stop + 2 * overlap, None
            )
            y_slice = subtile_selection["y"]
            subtile_selection["y"] = slice(
                y_slice.start, y_slice.stop + 2 * overlap, None
            )
        else:
            dataset_ = dataset

        dataset_.attrs.update(metadata)

        return dataset_.isel(subtile_selection)


class RankDivider:
    def __init__(
        self,
        subdomain_layout: Layout,
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
                subdomain_layout=(12, 12),
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

        self.x_ind = rank_dims.index("x")
        self.y_ind = rank_dims.index("y")

        self._partitioner = pace.util.TilePartitioner(subdomain_layout)
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]

        # dimensions of rank data without the halo points. Useful for slice calculation.
        self._rank_extent_without_overlap = self._get_rank_extent_without_overlap(
            rank_extent, overlap
        )

    def get_subdomain_extent(self, with_overlap: bool):
        subdomain_xy_size = (
            self._rank_extent_without_overlap[self.x_ind] // self.subdomain_layout[0]
        )
        if with_overlap:
            subdomain_xy_size += 2 * self.overlap

        subdomain_extent = list(self.rank_extent)
        subdomain_extent[self.x_ind] = subdomain_xy_size
        subdomain_extent[self.y_ind] = subdomain_xy_size
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
        x_slice_ = slice_[self.x_ind]
        y_slice_ = slice_[self.y_ind]

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

        slice_[self.x_ind] = x_slice_updated
        slice_[self.y_ind] = y_slice_updated
        return tuple(slice_)

    def _get_rank_extent_without_overlap(
        self, data_shape: Sequence[int], overlap: int
    ) -> Sequence[int]:
        extent_without_halos = list(data_shape)
        extent_without_halos[self.x_ind] = (
            extent_without_halos[self.x_ind] - 2 * overlap
        )
        extent_without_halos[self.y_ind] = (
            extent_without_halos[self.y_ind] - 2 * overlap
        )
        return tuple(extent_without_halos)

    def get_subdomain_tensor_slice(
        self, tensor_data: tf.Tensor, subdomain_index: int, with_overlap: bool
    ) -> tf.Tensor:

        subdomain_slice = self.subdomain_slice(subdomain_index, with_overlap)
        tensor_data_xsliced = slice_along_axis(
            arr=tensor_data, inds=subdomain_slice[self.x_ind], axis=self.x_ind
        )
        tensor_data_xy_sliced = slice_along_axis(
            arr=tensor_data_xsliced, inds=subdomain_slice[self.y_ind], axis=self.y_ind
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

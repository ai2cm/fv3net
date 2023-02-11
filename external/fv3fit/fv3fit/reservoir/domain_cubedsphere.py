import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple
import xarray as xr

import pace.util
from fv3fit.keras._models.shared.halos import append_halos

Layout = Tuple[int, int]


def _slice(arr: np.ndarray, inds: slice, axis: int = 0):
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

    def get_rank_data(self, dataset: xr.Dataset, rank: int, overlap: int):

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
        self._n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._rank_extent_without_overlap = self._get_extent_without_overlap(
            rank_extent, overlap
        )

    def _get_subdomain_extent(self, with_overlap: bool):
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
        # first get the slices w/o overlap points on XY data that does not have halo
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
            x_slice_updated = slice(
                x_slice_.start + self.overlap, x_slice_.stop + self.overlap, None
            )
            y_slice_updated = slice(
                y_slice_.start + self.overlap, y_slice_.stop + self.overlap, None
            )

        slice_[self.x_ind] = x_slice_updated
        slice_[self.y_ind] = y_slice_updated
        return tuple(slice_)

    def _get_extent_without_overlap(
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
    ):

        subdomain_slice = self.subdomain_slice(subdomain_index, with_overlap)
        tensor_data_xsliced = _slice(
            arr=tensor_data, inds=subdomain_slice[self.x_ind], axis=self.x_ind
        )
        tensor_data_xy_sliced = _slice(
            arr=tensor_data_xsliced, inds=subdomain_slice[self.y_ind], axis=self.y_ind
        )
        return tensor_data_xy_sliced

    def unstack_subdomain(self, tensor, with_overlap: bool):
        unstacked_shape = self._get_subdomain_extent(with_overlap=with_overlap)[1:]
        expected_stacked_size = np.prod(unstacked_shape)
        if tensor.shape[-1] != expected_stacked_size:
            raise ValueError(
                "Dimension of each stacked sample expected to be "
                f"{expected_stacked_size} (product of {unstacked_shape})."
            )
        unstacked_shape = (tensor.shape[0], *unstacked_shape)
        return np.reshape(tensor, unstacked_shape)


def stack_time_series_samples(tensor):
    # assumes time is the first dimension
    n_samples = tensor.shape[0]
    return np.reshape(tensor, (n_samples, -1))

import numpy as np
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


class Subdomain:
    def __init__(self, data: np.ndarray, overlap: int, subdomain_axis: int = 0):
        self.overlapping = data
        self.overlap = overlap
        self.nonoverlapping = _slice(
            arr=data, inds=slice(overlap, -overlap), axis=subdomain_axis
        )


class PeriodicDomain:
    def __init__(
        self,
        data: np.ndarray,
        subdomain_size: int,
        subdomain_overlap: int,
        subdomain_axis: int = 0,
    ):
        self.data = data
        self.subdomain_size = subdomain_size
        if data.shape[subdomain_axis] % subdomain_size != 0:
            raise ValueError(f"Data size must be evenly divisible by subdomain_size")
        self.subdomain_overlap = subdomain_overlap
        self.subdomain_axis = subdomain_axis
        self.n_subdomains = data.shape[subdomain_axis] // subdomain_size
        self.index = 0

    def __len__(self) -> int:
        return self.n_subdomains

    def _pad_array_along_subdomain_axis(self, arr):
        n_dims = len(arr.shape)
        pad_widths = tuple(
            (0, 0)
            if axis != self.subdomain_axis
            else (self.subdomain_overlap, self.subdomain_overlap)
            for axis in range(n_dims)
        )
        return np.pad(arr, mode="wrap", pad_width=pad_widths)

    def __getitem__(self, index: int):
        padded = self._pad_array_along_subdomain_axis(self.data)
        start_ind = index * self.subdomain_size
        stop_ind = start_ind + self.subdomain_size + 2 * self.subdomain_overlap
        if index >= self.n_subdomains:
            raise ValueError(
                f"Cannot select subdomain with index {index}, there are "
                f"only {self.n_subdomains} subdomains."
            )
        subdomain_slice = _slice(
            arr=padded, inds=slice(start_ind, stop_ind), axis=self.subdomain_axis
        )
        return Subdomain(
            subdomain_slice,
            overlap=self.subdomain_overlap,
            subdomain_axis=self.subdomain_axis,
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_subdomains:
            raise StopIteration
        elem = self[self.index]
        self.index += 1
        return elem


class CubedsphereRankDivider:
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

    def get_rank_data(self, dataset: xr.Dataset, rank: int, overlap: int):
        subtile_slice = self.cubedsphere_partitioner.subtile_slice(
            rank,
            global_dims=self.global_dims,
            global_extent=self.global_extent,
            overlap=False,
        )
        subtile_selection = {
            dim: selection for (dim, selection) in zip(self.global_dims, subtile_slice)
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
        return dataset_.isel(subtile_selection)

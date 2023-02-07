from typing import Sequence, Tuple
import xarray as xr

import pace.util
from fv3fit.keras._models.shared.halos import append_halos

Layout = Tuple[int, int]


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
        self.total_ranks = self.cubedsphere_partitioner.total_ranks

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


# reuse tile paritioner where 'ranks' are now the subdomains
class RankDivider:
    def __init__(
        self,
        subdomain_layout: Layout,
        rank_dims: Sequence[str],
        rank_extent: Sequence[int],
    ):
        self.subdomain_layout = subdomain_layout
        self.rank_dims = rank_dims
        self.rank_extent = rank_extent
        self.partitioner = pace.util.TilePartitioner(subdomain_layout)
        self._n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._dims_without_overlap = {
            dim: extent for dim, extent in zip(rank_dims, rank_extent)
        }

    def subdomain_slice(self, subdomain_index: int):
        return self.partitioner.subtile_slice(
            rank=subdomain_index,
            global_dims=self.rank_dims,
            global_extent=self.rank_extent,
        )

    def _check_data_dims(self, data_dims, overlap):
        x_dims_mismatch = (
            data_dims["x"] != self._dims_without_overlap["x"] + 2 * overlap
        )
        y_dims_mismatch = (
            data_dims["y"] != self._dims_without_overlap["y"] + 2 * overlap
        )
        if x_dims_mismatch or y_dims_mismatch:
            raise ValueError(
                f"Data provided for partitioning must have x, y dimensions "
                f"({self._dims_without_overlap['x'] + 2 * overlap}, "
                f"{self._dims_without_overlap['x'] + 2 * overlap}) "
                "to include overlap grid cells."
            )

    def get_subdomain_data(
        self, dataset: xr.Dataset, subdomain_index: int, overlap: int
    ):
        # assumes the dataset provided already has halo points around it if overlap > 0
        self._check_data_dims(dataset.dims, overlap)

        subdomain_slice = self.subdomain_slice(subdomain_index)
        subdomain_selection = {
            dim: selection for (dim, selection) in zip(self.rank_dims, subdomain_slice)
        }
        if overlap > 0:
            x_slice = subdomain_selection["x"]
            subdomain_selection["x"] = slice(
                x_slice.start, x_slice.stop + 2 * overlap, None
            )
            y_slice = subdomain_selection["y"]
            subdomain_selection["y"] = slice(
                y_slice.start, y_slice.stop + 2 * overlap, None
            )
        return dataset.isel(subdomain_selection)

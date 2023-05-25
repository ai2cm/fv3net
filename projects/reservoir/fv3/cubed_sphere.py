from typing import Sequence, Tuple
import xarray as xr

import pace.util
from fv3fit._shared.halos import append_halos

Layout = Tuple[int, int]


class CubedSphereDivider:
    def __init__(
        self,
        tile_layout: Layout,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
    ):
        """ Class for dividing global cubed sphere data into ranks for
        preprocessing. Preprocessing before the training step saves the dataset
        in directories of netcdfs per rank where each .nc file is a chunk of that
        rank's time series data. This saves on loading time and memory usage
        during training, which is done for a single rank.

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

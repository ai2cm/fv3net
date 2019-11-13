import pathlib
# TODO: convenience and regrid are currently broken

# from .convenience import open_dataset
from .cubedsphere import (
    open_cubed_sphere, block_coarsen, block_edge_sum, block_median,
    xarray_block_reduce,
    edge_weighted_block_average, save_tiles_separately,
    horizontal_block_reduce,
    weighted_block_average_and_add_coarsened_subtile_coordinates,
    edge_weighted_block_average_and_add_coarsened_subtile_coordinates,
    horizontal_block_reduce_and_add_coarsened_subtile_coordinates,
    block_median_and_add_coarsened_subtile_coordinates,
    block_coarsen_and_add_coarsened_subtile_coordinates,
    block_edge_sum_and_add_coarsened_subtile_coordinates,
)
from .extract import extract_tarball_to_path
# from .regrid import regrid_horizontal

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()

__all__ = [
    # "open_dataset",
    "open_cubed_sphere",
    "block_coarsen",
    "block_edge_sum",
    "block_median",
    "edge_weighted_block_average",
    "save_tiles_separately",
    "extract_tarball_to_path",
    "xarray_block_reduce",
    "horizontal_block_reduce",
    "weighted_block_average_and_add_coarsened_subtile_coordinates",
    "edge_weighted_block_average_and_add_coarsened_subtile_coordinates",
    "horizontal_block_reduce_and_add_coarsened_subtile_coordinates",
    "block_median_and_add_coarsened_subtile_coordinates",
    "block_coarsen_and_add_coarsened_subtile_coordinates",
    "block_edge_sum_and_add_coarsened_subtile_coordinates",
    # "regrid_horizontal",
    "TOP_LEVEL_DIR",
]

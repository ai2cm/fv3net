import pathlib

from .combining import combine_array_sequence

# from .convenience import open_dataset
from .cubedsphere import (
    block_coarsen,
    block_edge_sum,
    block_median,
    edge_weighted_block_average,
    horizontal_block_reduce,
    save_tiles_separately,
    xarray_block_reduce,
)
from .extract import extract_tarball_to_path
from .fv3_restarts import open_restarts

# TODO: convenience and regrid are currently broken


# from .regrid import regrid_horizontal

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()

__all__ = [
    "combine_array_sequence",
    "open_restarts",
    "block_coarsen",
    "block_edge_sum",
    "block_median",
    "edge_weighted_block_average",
    "save_tiles_separately",
    "extract_tarball_to_path",
    "xarray_block_reduce",
    "horizontal_block_reduce",
    "TOP_LEVEL_DIR",
]

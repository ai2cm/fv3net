import pathlib

# from .convenience import open_dataset
from .cubedsphere import (block_coarsen, block_edge_sum, block_median,
                          edge_weighted_block_average, horizontal_block_reduce,
                          open_cubed_sphere, save_tiles_separately,
                          xarray_block_reduce)
from .extract import extract_tarball_to_path

# TODO: convenience and regrid are currently broken


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
    # "regrid_horizontal",
    "TOP_LEVEL_DIR",
]

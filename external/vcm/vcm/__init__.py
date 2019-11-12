import pathlib

# TODO: convenience and regrid are currently broken

# from .convenience import open_dataset
from .cubedsphere import (
    open_cubed_sphere, block_coarsen, block_edge_sum, block_median, block_reduce,
    edge_weighted_block_average, save_tiles_separately
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
    "block_reduce",
    "edge_weighted_block_average",
    "save_tiles_separately",
    "extract_tarball_to_path",
    # "regrid_horizontal",
    "TOP_LEVEL_DIR",
]

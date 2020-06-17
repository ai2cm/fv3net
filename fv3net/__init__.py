import pathlib

__author__ = """Vulcan Technologies LLC"""
__version__ = "0.2.2"

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()

# TODO delete this hardcode
COARSENED_DIAGS_ZARR_NAME = "gfsphysics_15min_coarse.zarr"

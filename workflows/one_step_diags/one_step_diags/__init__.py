__version__ = "0.1.0"

from . import config
from .pipeline import run
from .config import INIT_TIME_DIM, FORECAST_TIME_DIM, OUTPUT_NC_FILENAME, ZARR_STEP_DIM

__all__ = [config, run]

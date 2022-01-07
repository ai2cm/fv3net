from typing import TypeVar
import numpy as np
import xarray as xr

Array = TypeVar("Array", np.ndarray, xr.DataArray, float)

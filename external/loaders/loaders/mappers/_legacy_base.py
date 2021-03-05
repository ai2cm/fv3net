import numpy as np
import pandas as pd
import xarray as xr
import vcm
from ._base import GeoMapper
from .._utils import standardize_zarr_time_coord
from ..constants import TIME_NAME, TIME_FMT


class LongRunMapper(GeoMapper):
    """
    Basic mapper across the time dimension for any long-form
    simulation output.
    
    This mapper uses slightly different initialization from the
    base GeoMapper class (this takes a dataset instead of a url) because
    run information for all timesteps already exists within
    a single file, i.e., no filesystem grouping is necessary to get
    an item.
    """

    def __init__(self, ds: xr.Dataset):
        self.ds = standardize_zarr_time_coord(ds)

    def __getitem__(self, key: str) -> xr.Dataset:
        dt64 = np.datetime64(vcm.parse_datetime_from_str(key))
        return self.ds.sel({TIME_NAME: dt64}).drop_vars(names=TIME_NAME)

    def keys(self):
        return set(
            [
                time.strftime(TIME_FMT)
                for time in pd.to_datetime(self.ds[TIME_NAME].values)
            ]
        )

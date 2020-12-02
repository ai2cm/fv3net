import pandas as pd
import numpy as np
import xarray as xr
import vcm
from typing import Hashable, Mapping, Optional, Sequence


from ..constants import DATASET_DIM_NAME, TIME_NAME, TIME_FMT
from .._utils import standardize_zarr_time_coord


class GeoMapper(Mapping[str, xr.Dataset]):
    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


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

    def __init__(self, ds):
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


class MultiDatasetMapper(GeoMapper):
    def __init__(
        self,
        mappers: Sequence[LongRunMapper],
        names: Optional[Sequence[Hashable]] = None,
    ):
        """Create a new MultiDatasetMapper.
        
        Args:
            mappers: sequence of LongRunMapper objects
            names: sequence of names to assign to the dataset coordinate (optional)
        """
        self.mappers = mappers
        self.names = names

    def keys(self):
        return set.intersection(*[set(mapper.keys()) for mapper in self.mappers])

    def __getitem__(self, time):
        if time not in self.keys():
            raise KeyError(f"Time {time} could not be found in all datasets.")
        else:
            datasets = [mapper[time] for mapper in self.mappers]
            if self.names is not None:
                dim = pd.Index(self.names, name=DATASET_DIM_NAME)
            else:
                dim = DATASET_DIM_NAME
            return xr.concat(datasets, dim=dim)


class XarrayMapper(GeoMapper):
    def __init__(self, data):
        """Create a mapper directly from an xarray dataset
        
        Args:
            data: xarray dataset with 'time' as a coordinate
        """
        self.data = data

        times = self.data.time.values.tolist()
        time_strings = [vcm.encode_time(time) for time in times]
        self.time_lookup = dict(zip(time_strings, times))
        self.time_string_lookup = dict(zip(times, time_strings))

    def __getitem__(self, time_string):
        return self.data.sel(time=self.time_lookup[time_string])

    def keys(self):
        return self.time_lookup.keys()

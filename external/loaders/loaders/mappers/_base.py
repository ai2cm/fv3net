import pandas as pd
import xarray as xr
import fsspec
import vcm
from typing import Hashable, Mapping, Optional, Sequence


from ..constants import DATASET_DIM_NAME, TIME_NAME


class GeoMapper(Mapping[str, xr.Dataset]):
    """Abstract base class for a mapper"""

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


class XarrayMapper(GeoMapper):
    """A mapper for accessing an xarray dataset.

    Example:
        >>> import cftime
        >>> data = xr.Dataset(
        ...         {"a": (["time", "x"], [[0, 0]])},
        ...         coords={"time": [cftime.DatetimeJulian(2016, 1, 1)]}
        ...        )
        >>> mapper = XarrayMapper(data, time="time")
        >>> list(mapper)
        ['20160101.000000']
        >>> mapper["20160101.000000"]
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
            time     object 2016-01-01 00:00:00
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 0 0
    """

    def __init__(self, ds: xr.Dataset, time: str = TIME_NAME):
        """Create an XarrayMapper
        
        Args:
            ds (xr.Dataset):
                Dataset, assumed to have datetime-like time coordinate
            time (str):
                name of dataset time coordinate
        """
        self.ds = ds

        times = self.ds[time].values.tolist()
        time_strings = [vcm.encode_time(single_time) for single_time in times]
        self.time_lookup = dict(zip(time_strings, times))
        self.time_string_lookup = dict(zip(times, time_strings))

    def __getitem__(self, time_string):
        return self.ds.sel(time=self.time_lookup[time_string])

    def keys(self):
        """The mapper's time keys"""
        return self.time_lookup.keys()


class MultiDatasetMapper(GeoMapper):
    def __init__(
        self,
        mappers: Sequence[XarrayMapper],
        names: Optional[Sequence[Hashable]] = None,
    ):
        """Create a new MultiDatasetMapper.
        
        Args:
            mappers: sequence of XarrayMapper objects
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


def open_zarr(url: str, consolidated: bool = True, dim: str = "time") -> XarrayMapper:
    """Open a zarr and return a XarrayMapper
    
    Args:
        url (str):
            location of zarr store
        consolidate (bool):
            whether to open zarr with consolidated metadata; defaults to True
        dim (str):
            name of time dimension; defaults to 'time'
            
    Returns:
        XarrayMapper
    
    """
    ds = xr.open_zarr(fsspec.get_mapper(url), consolidated=consolidated)
    return XarrayMapper(ds, dim)

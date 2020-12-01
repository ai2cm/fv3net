import xarray as xr
import vcm
from ._base import GeoMapper


class XarrayMapper(GeoMapper):
    """A mapper for accessing an xarray dataset

    Example:
        >>> import cftime
        >>> data = xr.Dataset({"a": (["time", "x"], [[0, 0]])}, coords={"time": [cftime.DatetimeJulian(2016, 1, 1)]})
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
    def __init__(self, data: xr.Dataset, time: str = "time"):
        """

        Args:
            data: the xarray dataset to wrap
            time: the time dimension to access the data along. Must contain
                cftime.DatetimeJulian or datetime.datetime objects.

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

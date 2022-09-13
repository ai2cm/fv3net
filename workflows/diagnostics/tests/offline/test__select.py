import pytest
import cftime
import xarray as xr
from fv3net.diagnostics.offline._select import select_snapshot


BATCH_ONE = [
    cftime.DatetimeJulian(2016, 8, 1, 11),
    cftime.DatetimeJulian(2016, 8, 1, 14),
]
BATCH_TWO = [
    cftime.DatetimeJulian(2016, 8, 1, 17),
    cftime.DatetimeJulian(2016, 8, 1, 20),
]
BATCH_THREE = [
    cftime.DatetimeJulian(2016, 8, 1, 11, 30),
    cftime.DatetimeJulian(2016, 8, 1, 20),
]


ORDERED_TIMES = [
    cftime.DatetimeJulian(2000, 1, 1),
    cftime.DatetimeJulian(2000, 1, 7),
    cftime.DatetimeJulian(2000, 1, 15),
]
UNORDERED_TIMES = [
    cftime.DatetimeJulian(2000, 1, 7),
    cftime.DatetimeJulian(2000, 1, 15),
    cftime.DatetimeJulian(2000, 1, 1),
]


@pytest.mark.parametrize(
    "times", [ORDERED_TIMES, UNORDERED_TIMES], ids=["ordered-times", "unordered-times"]
)
def test_select_snapshot(times):
    batch = xr.Dataset(data_vars={"foo": (["time"], [0, 1, 2])}, coords={"time": times})
    result = select_snapshot(batch, cftime.DatetimeJulian(2000, 1, 5))
    expected = batch.sel(time=cftime.DatetimeJulian(2000, 1, 7))
    xr.testing.assert_identical(result, expected)

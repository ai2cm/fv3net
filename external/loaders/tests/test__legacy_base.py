import pytest
import xarray as xr
import pandas as pd
from loaders.mappers._nudged._legacy import LongRunMapper
from loaders import TIME_NAME, TIME_FMT


def construct_dataset(num_tsteps):
    xdim = 10
    coords = {
        TIME_NAME: pd.date_range("2000-01-01", periods=num_tsteps),
        "x": range(xdim),
    }
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [[x for x in range(xdim)] for t in range(num_tsteps)],
        dims=["time", "x"],
        coords=coords,
    )
    return xr.Dataset({"var": var})


@pytest.fixture
def ds(request):
    return construct_dataset(request.param)


@pytest.mark.parametrize("ds", [1, 5], indirect=True)
def test_LongRunMapper(ds):
    mapper = LongRunMapper(ds)

    assert len(mapper) == ds.sizes[TIME_NAME]

    single_time = ds[TIME_NAME].values[0]
    item = ds.sel({TIME_NAME: single_time}).drop_vars(names=TIME_NAME)
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    xr.testing.assert_equal(item, mapper[time_key])

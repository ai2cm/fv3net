from datetime import datetime
import pandas as pd
import pytest
import xarray as xr
from loaders import TIME_NAME, TIME_FMT
from loaders.mappers import LongRunMapper


@pytest.fixture
def ds(request):
    num_tsteps = request.param
    xdim = 10
    coords = {
        TIME_NAME: [
            datetime.strptime(f"{20000100+int(t+1)}.000000", TIME_FMT)
            for t in range(num_tsteps)
        ],
        "x": range(xdim),
    }
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [[x for x in range(xdim)] for t in range(num_tsteps)],
        dims=["time", "x"],
        coords=coords,
    )
    return xr.Dataset({"var": var})


@pytest.mark.parametrize("ds", [1, 5], indirect=True)
def test_LongRunMapper(ds):
    mapper = LongRunMapper(ds)

    assert len(mapper) == ds.sizes[TIME_NAME]

    single_time = ds[TIME_NAME].values[0]
    item = ds.sel({TIME_NAME: single_time})
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    xr.testing.assert_equal(item, mapper[time_key])

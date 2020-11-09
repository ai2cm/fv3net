import pandas as pd
import pytest
import xarray as xr
from loaders import DATASET_DIM_NAME, TIME_NAME, TIME_FMT
from loaders.mappers import LongRunMapper, MultiDatasetMapper


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


@pytest.mark.parametrize(
    "sizes", [(1, 1), (1, 2), (3, 2), (2, 3, 5), (1, 1, 3)], ids=lambda x: f"sizes={x}"
)
def test_MultiDatasetMapper(sizes):
    datasets = [construct_dataset(size) for size in sizes]
    mappers = [LongRunMapper(ds) for ds in datasets]
    mapper = MultiDatasetMapper(mappers)
    expected_length = min(sizes)
    expected_keys = set(
        pd.date_range("2000-01-01", periods=expected_length).strftime(TIME_FMT)
    )

    assert len(mapper) == expected_length
    assert mapper.keys() == expected_keys

    single_time = datasets[0][TIME_NAME].isel({TIME_NAME: 0}).item()
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    expected_ds = xr.concat(
        [
            ds.sel({TIME_NAME: single_time}).drop_vars(names=TIME_NAME)
            for ds in datasets
        ],
        dim=DATASET_DIM_NAME,
    )
    xr.testing.assert_identical(expected_ds, mapper[time_key])

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


@pytest.fixture(
    params=[(1, 1), (1, 2), (3, 2), (2, 3, 5), (1, 1, 3)], ids=lambda x: f"sizes={x}"
)
def sizes(request):
    return request.param


@pytest.fixture
def expected_length(sizes):
    return min(sizes)


@pytest.fixture
def expected_keys(expected_length):
    return set(pd.date_range("2000-01-01", periods=expected_length).strftime(TIME_FMT))


@pytest.fixture
def datasets(sizes):
    return [construct_dataset(size) for size in sizes]


@pytest.fixture
def multi_dataset_mapper(datasets):
    mappers = [LongRunMapper(ds) for ds in datasets]
    return MultiDatasetMapper(mappers)


def test_MultiDatasetMapper_length(multi_dataset_mapper, expected_length):
    assert len(multi_dataset_mapper) == expected_length


def test_MultiDatasetMapper_keys(multi_dataset_mapper, expected_keys):
    assert multi_dataset_mapper.keys() == expected_keys


def test_MultiDatasetMapper_value(multi_dataset_mapper, datasets):
    single_time = datasets[0][TIME_NAME].isel({TIME_NAME: 0}).item()
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    expected_dataset = xr.concat(
        [
            ds.sel({TIME_NAME: single_time}).drop_vars(names=TIME_NAME)
            for ds in datasets
        ],
        dim=DATASET_DIM_NAME,
    )
    xr.testing.assert_identical(multi_dataset_mapper[time_key], expected_dataset)

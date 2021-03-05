import string
import doctest
import numpy as np
import pytest
import xarray as xr
import cftime
from loaders import DATASET_DIM_NAME, TIME_NAME, TIME_FMT
from loaders.mappers import XarrayMapper, open_zarr
from loaders.mappers._base import MultiDatasetMapper
import loaders.mappers._base


def test_xarray_wrapper_doctests():
    doctest.testmod(loaders.mappers._base, raise_on_error=True)


def construct_dataset(num_tsteps):
    xdim = 10
    time_coord = [cftime.DatetimeJulian(2000, 1, 1 + t) for t in range(num_tsteps)]
    coords = {
        TIME_NAME: time_coord,
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
def test_XarrayMapper(ds):
    mapper = XarrayMapper(ds)

    assert len(mapper) == ds.sizes[TIME_NAME]

    single_time = ds[TIME_NAME].values[0]
    item = ds.sel({TIME_NAME: single_time})
    time_key = single_time.strftime(TIME_FMT)
    xr.testing.assert_equal(item, mapper[time_key])


def test_open_zarr(tmpdir):
    time = cftime.DatetimeJulian(2020, 1, 1)
    time_str = "20200101.000000"
    ds = xr.Dataset(
        {"a": (["time", "tile", "z", "y", "x"], np.ones((1, 2, 3, 4, 5)))},
        coords={"time": [time]},
    )
    ds.to_zarr(str(tmpdir), consolidated=True)

    mapper = open_zarr(str(tmpdir))
    assert isinstance(mapper, XarrayMapper)
    xr.testing.assert_equal(mapper[time_str], ds.isel({TIME_NAME: 0}))


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
    return set(
        [
            cftime.DatetimeJulian(2000, 1, 1 + t).strftime(TIME_FMT)
            for t in range(expected_length)
        ]
    )


@pytest.fixture
def datasets(sizes):
    return [construct_dataset(size) for size in sizes]


@pytest.fixture
def multi_dataset_mapper(datasets):
    mappers = [XarrayMapper(ds) for ds in datasets]
    return MultiDatasetMapper(mappers)


def test_MultiDatasetMapper_len(multi_dataset_mapper, expected_length):
    assert len(multi_dataset_mapper) == expected_length


def test_MultiDatasetMapper_keys(multi_dataset_mapper, expected_keys):
    assert multi_dataset_mapper.keys() == expected_keys


def test_MultiDatasetMapper_value(multi_dataset_mapper, datasets):
    single_time = datasets[0][TIME_NAME].isel({TIME_NAME: 0}).item()
    time_key = single_time.strftime(TIME_FMT)
    expected_dataset = xr.concat(
        [ds.sel({TIME_NAME: single_time}) for ds in datasets], dim=DATASET_DIM_NAME,
    )
    xr.testing.assert_identical(multi_dataset_mapper[time_key], expected_dataset)


def test_MultiDatasetMapper_key_error(multi_dataset_mapper):
    with pytest.raises(KeyError, match="all datasets"):
        multi_dataset_mapper["20000103.000000"]


@pytest.fixture
def multi_dataset_mapper_with_names(datasets):
    mappers = [XarrayMapper(ds) for ds in datasets]
    names = [i for i, _ in zip(string.ascii_lowercase, datasets)]
    return MultiDatasetMapper(mappers, names=names)


def test_multidataset_mapper_with_names(datasets, multi_dataset_mapper_with_names):
    single_time = datasets[0][TIME_NAME].isel({TIME_NAME: 0}).item()
    time_key = single_time.strftime(TIME_FMT)
    assert "a" in multi_dataset_mapper_with_names[time_key][DATASET_DIM_NAME]

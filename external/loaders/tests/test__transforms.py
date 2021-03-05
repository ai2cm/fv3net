from loaders.mappers import ValMap, KeyMap, SubsetTimes, XarrayMapper
import xarray as xr
import numpy as np
import cftime

import pytest


# test components
item = xr.Dataset({"a": (["x"], [1.0])})
original_mapper = {"a": item}


def func(x):
    return 2 * x


val_mapper = ValMap(func, original_mapper)
key_mapper = KeyMap(func, original_mapper)


def test_ValMap_raises_error_if_key_not_present():
    with pytest.raises(KeyError):
        val_mapper["not in here"]


def test_ValMap_correctly_applies_func():
    xr.testing.assert_equal(val_mapper["a"], func(item))


def test_ValMap_keys_unchanged():
    assert set(val_mapper) == set(original_mapper)


def test_KeyMap_value_unchanged():
    xr.testing.assert_equal(key_mapper[func("a")], original_mapper["a"])


def test_KeyMap_keys_changed():
    assert set(key_mapper) == set(func(key) for key in original_mapper)


TIME_DIM = 10
X_DIM = 5

time_coord = [cftime.DatetimeJulian(2020, 1, 1 + i) for i in range(TIME_DIM)]

ds = xr.Dataset(
    {"a": xr.DataArray(np.ones([TIME_DIM, X_DIM]), dims=["time", "x"])},
    coords={"time": time_coord, "x": np.arange(X_DIM)},
)


@pytest.fixture
def mapper():
    return XarrayMapper(ds)


def test_SubsetTime(mapper):

    i_start = 4
    n_times = 6
    times = sorted(list(mapper.keys()))[4:10]

    subset = SubsetTimes(i_start, n_times, mapper)

    assert len(subset) == n_times
    assert times == sorted(list(subset.keys()))


def test_SubsetTime_out_of_order_times(mapper):

    times = sorted(list(mapper.keys()))[:5]
    shuffled_idxs = [4, 0, 2, 3, 1]
    shuffled_map = {times[i]: mapper[times[i]] for i in shuffled_idxs}
    subset = SubsetTimes(0, 2, shuffled_map)

    for i, key in enumerate(sorted(list(subset.keys()))):
        assert key == times[i]
        xr.testing.assert_equal(mapper[key], subset[key])


def test_SubsetTime_fail_on_non_subset_key(mapper):

    out_of_bounds = sorted(list(mapper.keys()))[4]
    subset = SubsetTimes(0, 4, mapper)

    with pytest.raises(KeyError):
        subset[out_of_bounds]

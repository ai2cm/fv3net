from loaders.mappers import ValMap, KeyMap
import xarray as xr

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

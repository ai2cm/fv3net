import numpy as np
import xarray as xr
from generate_movie_stills import _movie_specs, _non_zero


def test__movie_specs():
    movie_specs = _movie_specs()
    for name, spec in movie_specs.items():
        func = spec["plotting_function"]
        variables = spec["required_variables"]
        assert callable(func)
        assert isinstance(variables, list)


def test__non_zero():
    da_zeros = xr.DataArray(np.zeros(5))
    da_not_zeros = xr.DataArray(np.ones(5))
    assert not _non_zero(xr.Dataset({"a": da_zeros}), ["a"])
    assert _non_zero(xr.Dataset({"b": da_not_zeros}), ["b"])
    assert not _non_zero(xr.Dataset({"a": da_zeros, "b": da_not_zeros}), ["a"])
    assert not _non_zero(xr.Dataset({"a": da_zeros}), ["b"])
    assert _non_zero(xr.Dataset({"a": da_zeros, "b": da_not_zeros}), ["a", "b"])

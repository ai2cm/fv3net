import numpy as np
import xarray as xr
import pytest
from typing import Sequence, Tuple

from fv3net.diagnostics._shared.registry import Registry


def _get_registry():
    def merge(diags: Sequence[Tuple[str, xr.Dataset]]) -> xr.Dataset:
        out = {}
        for suffix, ds in diags:
            for variable in ds:
                out[f"{variable}_{suffix}"] = ds[variable]
        return xr.Dataset(out)

    return Registry(merge)


def test_registry_compute():
    da = xr.DataArray(np.reshape(np.arange(20), (4, 5)), dims=["time", "x"])
    ds = xr.Dataset({"wind": da, "temperature": da})
    registry = _get_registry()

    @registry.register("time_mean")
    def compute_mean(data, dim="time"):
        return data.mean(dim)

    @registry.register("x_max")
    def compute_max(data, dim="x"):
        return data.max(dim)

    output = registry.compute(ds, n_jobs=1)
    expected_output = xr.Dataset(
        {
            "wind_time_mean": ds.wind.mean("time"),
            "temperature_time_mean": ds.temperature.mean("time"),
            "wind_x_max": ds.wind.max("x"),
            "temperature_x_max": ds.temperature.max("x"),
        }
    )
    xr.testing.assert_identical(output, expected_output)


def test_registry_raises_value_error():
    registry = _get_registry()

    @registry.register("time_mean")
    def compute_mean(data, dim="time"):
        return data.mean(dim)

    with pytest.raises(ValueError):

        @registry.register("time_mean")
        def compute_mean_again(data, dim="time"):
            return data.mean(dim)

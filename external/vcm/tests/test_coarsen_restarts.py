import os

import fsspec
import pytest
import vcm
import xarray as xr

from vcm.cubedsphere.coarsen_restarts import (
    coarsen_restarts_on_sigma,
    coarsen_restarts_on_pressure,
    coarsen_restarts_via_blended_method,
)


FACTOR = 2
RESTART_CATEGORIES = ["fv_core.res", "fv_tracer.res", "fv_srf_wnd.res", "sfc_data"]
REFERENCE_DATA = "gs://vcm-ml-code-testing-data/coarsen-restarts-reference-data"
FINE_DATA = os.path.join(REFERENCE_DATA, "C12")
COARSE_DATA = os.path.join(REFERENCE_DATA, "C6")
REGRESSION_TESTS = [
    (coarsen_restarts_on_sigma, {"coarsen_agrid_winds": True, "mass_weighted": True}),
    (coarsen_restarts_on_sigma, {"coarsen_agrid_winds": False, "mass_weighted": False}),
    (coarsen_restarts_on_pressure, {"coarsen_agrid_winds": True}),
    (coarsen_restarts_on_pressure, {"coarsen_agrid_winds": False}),
    (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": True, "mass_weighted": True},
    ),
    (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": False, "mass_weighted": False},
    ),
]


def regression_test_tag(func, kwargs):
    kwargs_tag = "--".join([f"{k}-{v}" for k, v in kwargs.items()])
    return f"{func.__name__}-{kwargs_tag}"


def reference_store(func, kwargs, category):
    tag = regression_test_tag(func, kwargs)
    return os.path.join(COARSE_DATA, f"{category}-{tag}.zarr")


def open_fine_restarts():
    data = {}
    for category in RESTART_CATEGORIES:
        stem = os.path.join(FINE_DATA, category)
        data[category] = vcm.open_tiles(stem)
    return data


def open_fine_grid():
    stem = os.path.join(FINE_DATA, "grid_spec")
    return vcm.open_tiles(stem)


def open_coarse_restarts(func, kwargs):
    data = {}
    for category in RESTART_CATEGORIES:
        store = reference_store(func, kwargs, category)
        data[category] = xr.open_zarr(fsspec.get_mapper(store))
    return data


@pytest.mark.parametrize(
    ("func", "kwargs"),
    REGRESSION_TESTS,
    ids=lambda x: x.__name__ if callable(x) else f"{x}",
)
def test_coarsen_restarts(func, kwargs):
    restarts = open_fine_restarts()
    grid = open_fine_grid()
    result = func(FACTOR, grid, restarts, **kwargs)
    result = {category: ds.compute() for category, ds in result.items()}
    expected = open_coarse_restarts(func, kwargs)

    for category in result:
        xr.testing.assert_allclose(result[category], expected[category])

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
REGRESSION_TESTS = {
    "mass-weighted-model-level-with-agrid-winds": (
        coarsen_restarts_on_sigma,
        {"coarsen_agrid_winds": True, "mass_weighted": True},
    ),
    "area-weighted-model-level-without-agrid-winds": (
        coarsen_restarts_on_sigma,
        {"coarsen_agrid_winds": False, "mass_weighted": False},
    ),
    "pressure-level-with-agrid-winds": (
        coarsen_restarts_on_pressure,
        {"coarsen_agrid_winds": True},
    ),
    "pressure-level-without-agrid-winds": (
        coarsen_restarts_on_pressure,
        {"coarsen_agrid_winds": False},
    ),
    "blended-mass-weighted-with-agrid-winds": (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": True},
    ),
    "blended-area-weighted-without-agrid-winds": (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": False, "mass_weighted": False},
    ),
}


def reference_store(tag, category):
    return os.path.join(COARSE_DATA, f"{tag}-{category}.zarr")


def open_fine_restarts():
    data = {}
    for category in RESTART_CATEGORIES:
        stem = os.path.join(FINE_DATA, category)
        data[category] = vcm.open_tiles(stem)
    return data


def open_fine_grid():
    stem = os.path.join(FINE_DATA, "grid_spec")
    return vcm.open_tiles(stem)


def open_coarse_restarts(tag):
    data = {}
    for category in RESTART_CATEGORIES:
        store = reference_store(tag, category)
        data[category] = xr.open_zarr(fsspec.get_mapper(store))
    return data


@pytest.mark.slow
@pytest.mark.parametrize("tag", REGRESSION_TESTS.keys())
def test_coarsen_restarts(tag):
    func, kwargs = REGRESSION_TESTS[tag]
    restarts = open_fine_restarts()
    grid = open_fine_grid()
    result = func(FACTOR, grid, restarts, **kwargs)
    result = {category: ds.compute() for category, ds in result.items()}
    expected = open_coarse_restarts(tag)

    for category in result:
        xr.testing.assert_allclose(result[category], expected[category])

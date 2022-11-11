import os

import joblib
import pytest
import vcm

from vcm.cubedsphere.coarsen_restarts import (
    coarsen_restarts_on_sigma,
    coarsen_restarts_on_pressure,
    coarsen_restarts_via_blended_method,
)


FACTOR = 2
RESTART_CATEGORIES = ["fv_core.res", "fv_tracer.res", "fv_srf_wnd.res", "sfc_data"]
TEST_DATA = "gs://vcm-ml-code-testing-data/sample-c12-grid-and-restart-files"
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


def open_restarts():
    data = {}
    for category in RESTART_CATEGORIES:
        stem = os.path.join(TEST_DATA, category)
        data[category] = vcm.open_tiles(stem)
    return data


def open_grid():
    stem = os.path.join(TEST_DATA, "grid_spec")
    return vcm.open_tiles(stem)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    REGRESSION_TESTS,
    ids=lambda x: x.__name__ if callable(x) else f"{x}",
)
def test_coarsen_restarts(regtest, func, kwargs):
    restarts = open_restarts()
    grid = open_grid()
    result = func(FACTOR, grid, restarts, **kwargs)
    result = {category: ds.compute() for category, ds in result.items()}
    print(joblib.hash(result), file=regtest)

import numpy as np
import pytest
import xarray as xr
from vcm.cubedsphere.constants import COORD_X_CENTER, COORD_Y_CENTER
from vcm.select import mask_to_surface_type


@pytest.fixture()
def test_surface_type_grid():
    centered_coords = {COORD_Y_CENTER: [0], COORD_X_CENTER: [0, 1, 2]}
    slmsk_grid = xr.DataArray(
        [[0, 1, 2]], dims=[COORD_Y_CENTER, COORD_X_CENTER], coords=centered_coords
    )
    ds_slmsk = xr.Dataset({"slmsk": slmsk_grid, "checkvar": slmsk_grid})

    return ds_slmsk


def test_mask_to_surface_type(test_surface_type_grid):
    sea = mask_to_surface_type(test_surface_type_grid, "sea").checkvar.values
    np.testing.assert_equal(sea, np.array([[0, np.nan, np.nan]]))
    land = mask_to_surface_type(test_surface_type_grid, "land").checkvar.values
    np.testing.assert_equal(land, np.array([[np.nan, 1, np.nan]]))
    seaice = mask_to_surface_type(test_surface_type_grid, "seaice").checkvar.values
    np.testing.assert_equal(seaice, np.array([[np.nan, np.nan, 2]]))

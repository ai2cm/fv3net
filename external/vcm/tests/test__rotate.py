import numpy as np
import xarray as xr

from vcm.cubedsphere.rotate import rotate_xy_winds


def test_eastnorth_wind_tendencies():
    cell_centered_array = xr.DataArray(
        [[[1.0]], [[1.0]]],
        dims=["time", "x", "y"],
        coords={"time": [3, 4], "x": [0], "y": [0]},
    )
    expected_eastward_wind = xr.full_like(cell_centered_array, np.sqrt(2.0))
    expected_northward_wind = xr.full_like(cell_centered_array, 0.0)
    # lat/lon axes 45 deg rotated from x/y
    wind_rotation_matrix = xr.Dataset(
        {
            "eastward_wind_u_coeff": xr.DataArray(
                [[np.sqrt(2.0) / 2.0]], dims=["x", "y"], coords={"x": [0], "y": [0]}
            ),
            "eastward_wind_v_coeff": xr.DataArray(
                [[np.sqrt(2.0) / 2]], dims=["x", "y"], coords={"x": [0], "y": [0]}
            ),
            "northward_wind_u_coeff": xr.DataArray(
                [[-np.sqrt(2.0) / 2]], dims=["x", "y"], coords={"x": [0], "y": [0]}
            ),
            "northward_wind_v_coeff": xr.DataArray(
                [[np.sqrt(2.0) / 2]], dims=["x", "y"], coords={"x": [0], "y": [0]}
            ),
        }
    )
    eastward_wind, northward_wind = rotate_xy_winds(
        wind_rotation_matrix, cell_centered_array, cell_centered_array
    )
    xr.testing.assert_identical(eastward_wind, expected_eastward_wind)
    xr.testing.assert_identical(northward_wind, expected_northward_wind)

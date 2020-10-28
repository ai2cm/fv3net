import numpy as np
import xarray as xr

from vcm.cubedsphere.rotate import rotate_xy_winds


def test_eastnorth_wind_tendencies():
    cell_centered_array = xr.DataArray(
        [[1.0]], dims=["x", "y"], coords={"x": [0], "y": [0]}
    )
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
    result = rotate_xy_winds(
        wind_rotation_matrix, cell_centered_array, cell_centered_array
    )
    assert result[0] == [np.sqrt(2.0)]
    assert result[1] == [0.0]

import numpy as np
import xarray as xr

from vcm.cubedsphere.rotate import eastnorth_wind_tendencies


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
    result = eastnorth_wind_tendencies(
        wind_rotation_matrix, cell_centered_array, cell_centered_array
    )
    assert result["dQu"] == [np.sqrt(2.0)]
    assert result["dQv"] == [0.0]

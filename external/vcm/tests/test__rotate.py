import numpy as np
import pytest
import xarray as xr

from vcm.cubedsphere.rotate import (
    _wind_rotation_needed,
    _center_d_grid_winds,
    _eastnorth_wind_tendencies,
)


@pytest.mark.parametrize(
    "available_data_vars, result",
    (
        [["dQu", "dQv", "dQxwind", "dQywind"], False],
        [["dQ1", "dQ2", "dQxwind", "dQywind"], True],
        [["dQ1", "dQ2"], KeyError],
    ),
)
def test__wind_rotation_needed(available_data_vars, result):
    if result in [True, False]:
        assert _wind_rotation_needed(available_data_vars) == result
    else:
        with pytest.raises(KeyError):
            _wind_rotation_needed(available_data_vars)


cell_centered_array = xr.DataArray(
    [[1.0]], dims=["x", "y"], coords={"x": [0], "y": [0]}
)


def test__center_d_grid_winds():
    edge_xy_winds = xr.Dataset(
        {
            "dQxwind": xr.DataArray(
                [[2.0, 0.0]],
                dims=["x", "y_interface"],
                coords={"x": [0], "y_interface": [0, 1]},
            ),
            "dQywind": xr.DataArray(
                [[2.0, 0.0]],
                dims=["y", "x_interface"],
                coords={"y": [0], "x_interface": [0, 1]},
            ),
        }
    )
    centered_winds = _center_d_grid_winds(ds=edge_xy_winds)
    assert centered_winds["dQxwind"] == cell_centered_array
    assert centered_winds["dQywind"] == cell_centered_array


def test__eastnorth_wind_tendencies():
    ds = xr.Dataset({"dQxwind": cell_centered_array, "dQywind": cell_centered_array})
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
    result = _eastnorth_wind_tendencies(wind_rotation_matrix, ds=ds)
    assert result["dQu"] == [np.sqrt(2.0)]
    assert result["dQv"] == [0.0]

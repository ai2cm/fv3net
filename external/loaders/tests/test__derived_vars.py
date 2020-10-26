from datetime import datetime
import numpy as np
import pytest
import xarray as xr

from loaders.batches._derived_vars import (
    nonderived_variable_names,
    _wind_rotation_needed,
    _insert_cos_z,
    _center_d_grid_winds,
    _insert_latlon_wind_tendencies,
)


@pytest.mark.parametrize(
    "requested, available, nonderived_variables",
    (
        [["dQ1", "dQ2"], ["dQ1", "dQ2"], ["dQ1", "dQ2"]],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2"],
            ["dQ1", "dQ2", "dQxwind", "dQywind"],
        ],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2", "dQu", "dQv"],
            ["dQ1", "dQ2", "dQu", "dQv"],
        ],
    ),
)
def test_nonderived_variable_names(requested, available, nonderived_variables):
    assert set(nonderived_variable_names(requested, available)) == set(nonderived_variables)


@pytest.mark.parametrize(
    "available_data_vars, result",
    (
        [["dQu", "dQv", "dQx", "dQy"], False],
        [["dQ1", "dQ2", "dQx", "dQy"], True],
        [["dQ1", "dQ2"], KeyError],
    ),
)
def test__wind_rotation_needed(available_data_vars, result):
    xy = ["dQx", "dQy"]
    latlon = ["dQu", "dQv"]
    if result in [True, False]:
        assert _wind_rotation_needed(available_data_vars, xy, latlon) == result
    else:
        with pytest.raises(KeyError):
            _wind_rotation_needed(available_data_vars, xy, latlon)


def test__insert_cos_z():
    grid = xr.Dataset(
        {
            "lon": xr.DataArray([45.0], dims=["x"], coords={"x": [0]}),
            "lat": xr.DataArray([45.0], dims=["x"], coords={"x": [0]}),
        }
    )
    ds = xr.Dataset(
        {
            "var": xr.DataArray(
                [[0.0], [0.0]],
                dims=["time", "x"],
                coords={"time": [datetime(2020, 7, 6), datetime(2020, 7, 7)], "x": [0]},
            )
        }
    )
    cos_z_var = "cos_zenith_angle"
    ds = _insert_cos_z(grid, cos_z_var, ds)
    assert set(ds.data_vars) == {"var", cos_z_var}


cell_centered_array = xr.DataArray(
    [[1.0]], dims=["x", "y"], coords={"x": [0], "y": [0]}
)


def test__center_d_grid_winds():
    edge_xy_winds = xr.Dataset(
        {
            "dQx": xr.DataArray(
                [[2.0, 0.0]],
                dims=["x", "y_interface"],
                coords={"x": [0], "y_interface": [0, 1]},
            ),
            "dQy": xr.DataArray(
                [[2.0, 0.0]],
                dims=["y", "x_interface"],
                coords={"y": [0], "x_interface": [0, 1]},
            ),
        }
    )
    centered_winds = _center_d_grid_winds(
        xy_wind_tendency_vars=["dQx", "dQy"],
        latlon_wind_tendency_vars=["dQu", "dQv"],
        edge_to_center_dims={"x_interface": "x", "y_interface": "y"},
        ds=edge_xy_winds,
    )
    assert centered_winds["dQx"] == cell_centered_array
    assert centered_winds["dQy"] == cell_centered_array


@pytest.mark.parametrize(
    "data_vars, rotate", ([["dQx", "dQy"], True], [["dQu", "dQv"], False])
)
def test__insert_latlon_wind_tendencies(data_vars, rotate):
    ds = xr.Dataset({var: cell_centered_array for var in data_vars})
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
    result = _insert_latlon_wind_tendencies(
        wind_rotation_matrix,
        xy_wind_tendency_vars=["dQx", "dQy"],
        latlon_wind_tendency_vars=["dQu", "dQv"],
        ds=ds,
    )
    if rotate:
        assert result["dQu"] == [np.sqrt(2.0)]
        assert result["dQv"] == [0.0]
    else:
        assert result["dQu"] == [1.0]
        assert result["dQv"] == [1.0]

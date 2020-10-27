from datetime import datetime
import pytest
import xarray as xr

from loaders.batches._derived import (
    nonderived_variables,
    _insert_cos_z,
    _wind_rotation_needed,
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


@pytest.mark.parametrize(
    "requested, available, nonderived",
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
def test_nonderived_variable_names(requested, available, nonderived):
    assert set(nonderived_variables(requested, available)) == set(nonderived)


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
    ds = _insert_cos_z(grid, ds)
    assert set(ds.data_vars) == {"var", "cos_zenith_angle"}


cell_centered_array = xr.DataArray(
    [[1.0]], dims=["x", "y"], coords={"x": [0], "y": [0]}
)

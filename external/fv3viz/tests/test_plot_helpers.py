import numpy as np
import xarray as xr
import pytest

import fv3viz
from fv3viz._plot_helpers import (
    _get_var_label,
    _align_grid_var_dims,
    _align_plot_var_dims,
)


@pytest.mark.parametrize(
    "attrs,var_name,expected_label",
    [
        ({}, "temp", "temp"),
        ({"long_name": "air_temperature"}, "temp", "air_temperature"),
        ({"units": "degK"}, "temp", "temp [degK]"),
        (
            {"long_name": "air_temperature", "units": "degK"},
            "temp",
            "air_temperature [degK]",
        ),
    ],
)
def test__get_var_label(attrs, var_name, expected_label):
    assert _get_var_label(attrs, var_name) == expected_label


@pytest.mark.parametrize(
    "data, args, expected_result",
    [
        (np.array([0.0, 0.5, 1.0]), {}, (0.0, 1.0, "viridis")),
        (np.array([-0.5, 0, 1]), {}, (-1.0, 1.0, "RdBu_r")),
        (np.array([0.0, 0.5, 1.0]), {"robust": True}, (0.02, 0.98, "viridis")),
        (np.array([0.0, 0.5, 1.0]), {"cmap": "jet"}, (0.0, 1.0, "jet")),
        (np.array([0.0, 0.5, 1.0]), {"vmin": 0.2}, (0.2, 1.0, "viridis")),
        (np.array([-0.5, 0, 1]), {"vmin": -0.6}, (-0.6, 0.6, "RdBu_r")),
    ],
)
def test_infer_cmap_params(data, args, expected_result):
    result = fv3viz.infer_cmap_params(data, **args)
    assert result == expected_result


def get_var(dims, with_coords):
    if with_coords:
        coords = {
            dim: xr.DataArray(np.arange(4.0), dims=[dim], name=dim) for dim in dims
        }
    else:
        coords = {}
    grid_var = xr.DataArray(
        np.zeros([4 for _ in dims]), dims=dims, coords=coords, name="lat"
    )
    return grid_var


@pytest.mark.parametrize(
    ["grid_var_dims", "with_coords", "required_dims", "aligned_dims"],
    [
        pytest.param(
            ("x", "y", "tile"),
            True,
            ("x", "y", "tile"),
            ("x", "y", "tile"),
            id="identity",
        ),
        pytest.param(
            ("x", "y", "tile", "time"),
            True,
            ("x", "y", "tile"),
            ("x", "y", "tile"),
            id="extra dim with coords",
        ),
        pytest.param(
            ("x", "y", "tile", "time"),
            False,
            ("x", "y", "tile"),
            ("x", "y", "tile"),
            id="extra dim without coords",
        ),
        pytest.param(
            ("x", "y", "tile"),
            True,
            ("y", "x", "tile"),
            ("y", "x", "tile"),
            id="reorder dims",
        ),
    ],
)
def test__align_grid_var_dims(grid_var_dims, with_coords, required_dims, aligned_dims):
    grid_var = get_var(grid_var_dims, with_coords)
    grid_var_aligned = _align_grid_var_dims(grid_var, required_dims)
    assert grid_var_aligned.dims == aligned_dims


def test__align_grid_var_dims_raises_missing_dim():
    grid_var = get_var(("x", "tile"), True)
    with pytest.raises(ValueError):
        _align_grid_var_dims(grid_var, ("x", "y", "tile"))


@pytest.mark.parametrize(
    ["plot_var_dims", "coord_y_center", "coord_x_center", "aligned_dims"],
    [
        pytest.param(("y", "x", "tile"), "y", "x", ("y", "x", "tile"), id="identity"),
        pytest.param(
            ("grid_xt", "grid_yt", "tile"),
            "grid_yt",
            "grid_xt",
            ("grid_yt", "grid_xt", "tile"),
            id="reorder dims",
        ),
        pytest.param(
            ("y", "x", "extra_dim", "tile"),
            "y",
            "x",
            ("y", "x", "tile", "extra_dim"),
            id="extra dim",
        ),
    ],
)
def test__align_plot_var_dims(
    plot_var_dims, coord_y_center, coord_x_center, aligned_dims
):
    plot_var = get_var(plot_var_dims, False)
    plot_var_aligned = _align_plot_var_dims(plot_var, coord_y_center, coord_x_center)
    assert plot_var_aligned.dims == aligned_dims


def test__align_plot_var_dims_raises_missing_dim():
    plot_var = get_var(("x", "y"), False)
    with pytest.raises(ValueError):
        _align_plot_var_dims(plot_var, "x", "y")

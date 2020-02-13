import xarray as xr
import numpy as np
import pytest

from vcm.regrid import interpolate_unstructured, regrid_to_shared_coords


def test_interpolate_unstructured_same_as_sel_if_1d():
    n = 10
    ds = xr.Dataset({"a": (["x"], np.arange(n) ** 2)}, coords={"x": np.arange(n)})

    target_coords = {"x": xr.DataArray([5, 7], dims=["sample"])}

    output = interpolate_unstructured(ds, target_coords)
    expected = ds.sel(target_coords)

    np.testing.assert_equal(output, expected)


def _numpy_to_dataarray(x):
    return xr.DataArray(x, dims=["sample"])


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_interpolate_unstructured_2d(width):
    n = 3
    ds = xr.Dataset(
        {"a": (["j", "i"], np.arange(n * n).reshape((n, n)))},
        coords={"i": [-1, 0, 1], "j": [-1, 0, 1]},
    )

    # create a new coordinate system
    rotate_coords = dict(x=ds.i - ds.j, y=ds.i + ds.j)

    ds = ds.assign_coords(rotate_coords)

    # index all the data
    j = _numpy_to_dataarray(np.tile([-1, 0, 1], [3]))
    i = _numpy_to_dataarray(np.repeat([-1, 0, 1], 3))

    # get the rotated indices perturbed by noise
    eps = _numpy_to_dataarray(np.random.uniform(-width, width, size=n * n))
    x = i - j + eps
    y = i + j + eps

    # ensure that selecting with original coordinates is the same as interpolating
    # with the rotated ones
    expected = ds.sel(i=i, j=j)
    answer = interpolate_unstructured(ds, dict(x=x, y=y))

    xr.testing.assert_equal(expected["a"].variable, answer["a"].variable)
    assert expected["a"].dims == ("sample",)


@pytest.fixture()
def test_ds_interp():
    coords = {"pfull": [1, 2, 3], "x": [1, 2]}
    da_var_to_interp = xr.DataArray(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dims=["x", "pfull"], coords=coords
    )
    da_pressure = xr.DataArray(
        [[0, 1, 2], [0, 2, 4]], dims=["x", "pfull"], coords=coords
    )
    ds = xr.Dataset({"interp_var": da_var_to_interp, "pressure": da_pressure})
    return ds


def test_regrid_to_shared_coords(test_ds_interp):
    test_da = regrid_to_shared_coords(
        test_ds_interp["interp_var"],
        np.array([0.5, 2]),
        test_ds_interp["pressure"],
        "pressure_uniform",
        "pfull",
    )
    assert np.allclose(test_da.values, [[1.5, -1.25], [3.0, -2.0]])

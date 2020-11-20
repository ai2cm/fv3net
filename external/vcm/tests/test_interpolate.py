import xarray as xr
import numpy as np
import pytest

from vcm.interpolate import interpolate_unstructured, interpolate_1d, _interpolate_2d


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


def _test_dataset():
    coords = {"pfull": [1, 2, 3], "x": [1, 2]}
    da_var_to_interp = xr.DataArray(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dims=["x", "pfull"], coords=coords
    )
    da_pressure = xr.DataArray(
        [[0, 1, 2], [0, 2, 4]], dims=["x", "pfull"], coords=coords
    )
    ds = xr.Dataset({"interp_var": da_var_to_interp, "pressure": da_pressure})
    return ds


def test_interpolate_1d_dim_order_unchanged():

    ds = _test_dataset()

    output_pressure = xr.DataArray([0.5, 2], dims=["pressure_uniform"])

    test_da = interpolate_1d(
        output_pressure, ds["pressure"], ds["interp_var"], dim="pfull",
    )

    assert list(test_da.dims) == ["x", "pressure_uniform"]


def test_interpolate_1d_values_coords_correct():
    ds = _test_dataset()

    output_dim = "pressure_uniform"
    output_pressure = xr.DataArray([0.5, 2], dims=[output_dim])
    test_da = interpolate_1d(
        output_pressure, ds["pressure"], ds["interp_var"], "pfull",
    )

    expected = xr.DataArray([[1.5, 3.0], [-1.25, -2.0]], dims=["x", "pressure_uniform"])
    xr.testing.assert_allclose(test_da.variable, expected.variable)
    xr.testing.assert_allclose(test_da[output_dim].drop(output_dim), output_pressure)


def test_interpolate_1d_spatially_varying_levels():

    xp = xr.DataArray([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]], dims=["x", "y_new"])
    input_ = xr.DataArray([[0, 1], [2, 3]], dims=["x", "y"])
    x = xr.DataArray([[0, 1], [0, 1]], dims=["x", "y"])
    expected = xr.DataArray([[0.25, 0.5, 1.0], [2.25, 2.50, 3.0]], dims=["x", "y_new"])
    ans = interpolate_1d(xp, x, input_)
    xr.testing.assert_allclose(ans, expected)


def _interpolate_2d_reference(
    xp: np.ndarray, x: np.ndarray, y: np.ndarray, axis: int = 0
) -> np.ndarray:
    import scipy.interpolate

    output = np.zeros_like(xp, dtype=np.float64)
    for i in range(xp.shape[0]):
        output[i] = scipy.interpolate.interp1d(x[i], y[i], bounds_error=False)(xp[i])
    return output


def test__interpolate_2d():
    shape = (1, 10)
    new_shape = (1, 12)

    x = np.arange(10).reshape(shape)
    y = (x ** 2).reshape(shape)
    xp = np.arange(12).reshape(new_shape)

    expected = _interpolate_2d_reference(xp, x, y)
    assert np.isnan(expected[:, -2:]).all()

    ans = _interpolate_2d(xp, x, y)
    np.testing.assert_allclose(expected, ans)

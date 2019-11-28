import numpy as np
import pytest
import xarray as xr
from vcm.calc.diag_ufuncs import (
    apply_weighting,
    mean_over_dim,
    remove_extra_dim,
    sum_over_dim,
)


@pytest.fixture()
def test_ds_z():
    coords = {"pfull": [1, 2, 3], "forecast_time": [0]}
    da_diag_var = xr.DataArray(
        [[1.0, 2.0, 4.0]], dims=["forecast_time", "pfull"], coords=coords
    )
    da_delp = xr.DataArray([[4, 2, 2]], dims=["forecast_time", "pfull"], coords=coords)
    ds = xr.Dataset({"diag_var": da_diag_var, "delp": da_delp})
    return ds


@pytest.fixture()
def test_ds_xy():
    coords = {"x": [1, 2], "y": [1, 2]}
    da_diag_var = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=["x", "y"], coords=coords)
    da_weight_var = xr.DataArray([[4, 3], [2, 1]], dims=["x", "y"], coords=coords)
    ds = xr.Dataset({"diag_var": da_diag_var, "weight_var": da_weight_var})
    return ds


def test_remove_extra_dim(test_ds_z):
    ds = remove_extra_dim(test_ds_z, extra_dim="forecast_time")
    assert "forecast_time" not in ds.dims
    with pytest.raises(ValueError):
        remove_extra_dim(test_ds_z, extra_dim="pfull")


def test_apply_weighting(test_ds_xy):
    ds = apply_weighting(
        test_ds_xy,
        var_to_weight="diag_var",
        weighting_var="weight_var",
        weighting_dims=["x", "y"],
    )
    assert ds == pytest.approx(np.array([[0.4, 0.6], [0.6, 0.4]]))


def test_mean_over_dim(test_ds_xy):
    ds_x_mean = mean_over_dim(
        test_ds_xy, dim="x", var_to_avg="diag_var", new_var="x_mean_diag_var"
    )
    assert ds_x_mean["x_mean_diag_var"].values == pytest.approx([2, 3])


def test_sum_over_dim(test_ds_xy):
    ds_x_sum = sum_over_dim(
        test_ds_xy, dim="x", var_to_sum="diag_var", new_var="x_sum_diag_var"
    )
    assert ds_x_sum["x_sum_diag_var"].values == pytest.approx([4, 6])

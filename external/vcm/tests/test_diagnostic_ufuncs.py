import pytest
import xarray as xr
from vcm.calc.diag_ufuncs import(
    remove_extra_dim,
    apply_pressure_thickness_weighting,
    mean_over_dim
)

@pytest.fixture()
def test_ds():
    coords = {"pfull": [1, 2, 3], "forecast_time": [0]}
    da_diag_var = xr.DataArray([[1., 2., 4.]], dims=["forecast_time", "pfull"], coords=coords)
    da_delp = xr.DataArray([[4, 2, 2]], dims=["forecast_time", "pfull"], coords=coords)
    ds = xr.Dataset({"diag_var": da_diag_var, "delp": da_delp})
    return ds


def test_remove_extra_dim(test_ds):
    ds = remove_extra_dim(test_ds, extra_dim='forecast_time')
    assert 'forecast_time' not in ds.dims
    with pytest.raises(ValueError):
        remove_extra_dim(test_ds, extra_dim='pfull')


def test_apply_pressure_thickness_weighting(test_ds):
    ds = apply_pressure_thickness_weighting(test_ds, var_to_weight='diag_var')
    assert ds == pytest.approx([0.5, 0.5, 1])


def test_mean_over_dim(test_ds):
    ds_not_weighted = mean_over_dim(
        test_ds,
        dim='pfull',
        var_to_avg='diag_var',
        new_var='pres_weighted_diag_var',
        apply_delp_weighting=False)
    assert ds_not_weighted == pytest.approx([7./3])
    ds_weighted = mean_over_dim(
        test_ds,
        dim='pfull',
        var_to_avg='diag_var',
        new_var='pres_weighted_diag_var',
        apply_delp_weighting=True)
    assert ds_weighted == pytest.approx([])
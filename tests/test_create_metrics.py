import numpy as np
import pytest
import xarray as xr

from fv3net.diagnostics.sklearn_model_performance.create_metrics import (
    _rmse,
    _rmse_mass_avg
)


@pytest.fixture()
def test_ds():
    da_area = xr.DataArray(
        [[1., 4.]],
        dims=["y", "x"],
        coords={"x": [0, 1], "y": [0]},
    )
    da_pressure_thickness = xr.DataArray(
        [[[2., 1.], [3., 1.]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_target = xr.DataArray(
        [[[10., 20.], [-10., -20.]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )   
    da_pred = xr.DataArray(
        [[[11., 24.], [-13., -24.]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )   
    return xr.Dataset(
        {"area": da_area,
        "delp": da_pressure_thickness,
        "target": da_target,
        "pred": da_pred}
    )
    

def test__rmse(test_ds):
    rmse = _rmse(test_ds.target, test_ds.pred).isel({"x": 0, "y": 0})
    np.testing.assert_array_almost_equal(rmse, np.array([1., 4.]))


def test__rmse_mass_avg(test_ds):
    num2 = (1**2 * 2 + 4**2 *1) * 1. + (3**2 * 3 + 4**2 * 1) * 4.
    denom2 = (2+1)* 1. + (3+1)*4.
    rmse = _rmse_mass_avg(
        test_ds.target,
        test_ds.pred,
        test_ds.delp,
        test_ds.area,)
    assert rmse == pytest.approx(np.sqrt(num2/denom2))


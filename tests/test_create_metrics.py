import numpy as np
import pytest
import xarray as xr

from fv3net.diagnostics.sklearn_model_performance.create_metrics import (
    _bias,
    _rmse,
    _rmse_mass_avg,
)


@pytest.fixture()
def test_ds():
    da_area = xr.DataArray(
        [[1.0, 4.0]], dims=["y", "x"], coords={"x": [0, 1], "y": [0]}
    )
    da_pressure_thickness = xr.DataArray(
        [[[2.0, 1.0], [3.0, 1.0]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_target = xr.DataArray(
        [[[10.0, 20.0], [-10.0, -20.0]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_pred = xr.DataArray(
        [[[11.0, 24.0], [-13.0, -24.0]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    return xr.Dataset(
        {
            "area": da_area,
            "delp": da_pressure_thickness,
            "target": da_target,
            "pred": da_pred,
        }
    )


def test__bias(test_ds):
    bias = _bias(test_ds.target, test_ds.pred)
    assert bias == pytest.approx((1.0 + 4.0 - 3.0 - 4.0) / 4)

    area_weights = test_ds.area / test_ds.area.mean()  # [[0.4, 1.6]]
    weighted_bias = _bias(test_ds.target, test_ds.pred, weights=area_weights)
    assert weighted_bias == pytest.approx(
        (1.0 * 0.4 + 4.0 * 0.4 - 3.0 * 1.6 - 4.0 * 1.6) / 4
    )


def test__rmse(test_ds):
    rmse = _rmse(
        test_ds.target.isel({"x": 0, "y": 0}),
        test_ds.pred.isel({"x": 0, "y": 0}),
        mean_dims=["z"],
    )
    assert rmse == pytest.approx(np.sqrt((1 + 16.0) / 2))

    area_weights = test_ds.area / test_ds.area.mean()  # [[0.4, 1.6]]
    weighted_rmse = _rmse(
        test_ds.target.isel(z=0), test_ds.pred.isel(z=0), weights=area_weights
    )
    assert weighted_rmse == pytest.approx(np.sqrt((1 * 0.4 + 9.0 * 1.6) / 2.0))


def test__rmse_mass_avg(test_ds):
    num2 = (1 ** 2 * 2 + 4 ** 2 * 1) * 1.0 + (3 ** 2 * 3 + 4 ** 2 * 1) * 4.0
    denom2 = (2 + 1) * 1.0 + (3 + 1) * 4.0
    rmse = _rmse_mass_avg(
        test_ds.target,
        test_ds.pred,
        test_ds.delp,
        test_ds.area,
        coords_horizontal=["x", "y"],
    )
    assert rmse == pytest.approx(np.sqrt(num2 / denom2))

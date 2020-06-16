import numpy as np
import pytest
from typing import Mapping
import xarray as xr
from offline_ml_diags._metrics import (
    _bias,
    _rmse,
    _calc_batch_metrics,
    calc_metrics,
    DATA_SOURCE_DIM,
    TARGET_COORD,
    PREDICT_COORD,
)


@pytest.fixture()
def ds_mock():
    da = xr.DataArray(
        [[0.0, 1.0]],
        dims=["z", DATA_SOURCE_DIM],
        coords={"z": [0], DATA_SOURCE_DIM: [TARGET_COORD, PREDICT_COORD]},
    )
    delp = xr.DataArray([1.0], dims=["z"], coords={"z": [0]})
    return xr.Dataset(
        {"dQ1": da, "dQ2": da, "pressure_thickness_of_atmospheric_layer": delp}
    )


def test__calc_batch_metrics(ds_mock):
    batch_metrics = _calc_batch_metrics(ds_mock)
    for var in list(batch_metrics.data_vars):
        assert isinstance(batch_metrics[var].values.item(), float)


def test__calc_metrics(ds_mock):
    ds_sequence = [ds_mock, ds_mock]
    metrics_dict = calc_metrics(ds_sequence)
    assert isinstance(metrics_dict, Mapping)


@pytest.fixture()
def ds_calc_test(request):
    target_data, pred_data, area = request.param
    da_area = xr.DataArray(area, dims=["y", "x"], coords={"x": [0, 1], "y": [0]})
    da_pressure_thickness = xr.DataArray(
        [[[2.0, 1.0], [3.0, 1.0]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_target = xr.DataArray(
        target_data, dims=["y", "x", "z"], coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_pred = xr.DataArray(
        pred_data, dims=["y", "x", "z"], coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    return xr.Dataset(
        {
            "area": da_area,
            "delp": da_pressure_thickness,
            "target": da_target,
            "pred": da_pred,
        }
    )


@pytest.mark.parametrize(
    "ds_calc_test, target_unweighted_bias, target_weighted_bias",
    (
        (
            [
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[1.0, 4.0]],
            ],
            (1.0 + 4.0 - 3.0 - 4.0) / 4,
            (1.0 * 0.4 + 4.0 * 0.4 - 3.0 * 1.6 - 4.0 * 1.6) / 4,
        ),
        (
            [
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[1.0, 4.0]],
            ],
            0.0,
            0.0,
        ),
        (
            [
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[4.0, 4.0]],
            ],
            (1.0 + 4.0 - 3.0 - 4.0) / 4,
            (1.0 + 4.0 - 3.0 - 4.0) / 4,
        ),
        (
            [
                [[[np.nan, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[4.0, 4.0]],
            ],
            (4.0 - 3.0 - 4.0) / 3,
            (4.0 - 3.0 - 4.0) / 3,
        ),
    ),
    indirect=["ds_calc_test"],
)
def test__bias(ds_calc_test, target_unweighted_bias, target_weighted_bias):
    bias = _bias(ds_calc_test.target, ds_calc_test.pred)
    assert bias.values == pytest.approx(target_unweighted_bias)

    area_weights = ds_calc_test.area / ds_calc_test.area.mean()
    weighted_bias = _bias(ds_calc_test.target, ds_calc_test.pred, weights=area_weights)
    assert weighted_bias.values == pytest.approx(target_weighted_bias)


@pytest.mark.parametrize(
    "ds_calc_test, target_unweighted_rmse, target_weighted_rmse",
    (
        (
            [
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[4.0, 4.0]],
            ],
            np.sqrt((1.0 + 16.0 + 9.0 + 16.0) / 4.0),
            np.sqrt((1.0 + 16.0 + 9.0 + 16.0) / 4.0),
        ),
        (
            [
                [[[10.0, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[1.0, 4.0]],
            ],
            np.sqrt((1.0 + 16.0 + 9.0 + 16.0) / 4.0),
            np.sqrt((1.0 * 0.4 + 16.0 * 0.4 + 9.0 * 1.6 + 16.0 * 1.6) / 4.0),
        ),
        (
            [
                [[[np.nan, 20.0], [-10.0, -20.0]]],
                [[[11.0, 24.0], [-13.0, -24.0]]],
                [[4.0, 4.0]],
            ],
            np.sqrt((16.0 + 9.0 + 16.0) / 3.0),
            np.sqrt((16.0 + 9.0 + 16.0) / 3.0),
        ),
    ),
    indirect=["ds_calc_test"],
)
def test__rmse(ds_calc_test, target_unweighted_rmse, target_weighted_rmse):
    unweighted_rmse = _rmse(ds_calc_test.target, ds_calc_test.pred)
    assert unweighted_rmse == pytest.approx(target_unweighted_rmse)
    area_weights = ds_calc_test.area / ds_calc_test.area.mean()
    weighted_rmse = _rmse(ds_calc_test.target, ds_calc_test.pred, weights=area_weights)
    assert weighted_rmse == pytest.approx(target_weighted_rmse)

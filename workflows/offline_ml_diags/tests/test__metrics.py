import numpy as np
import pytest
import xarray as xr
from offline_ml_diags._metrics import (
    _weighted_average,
    _calc_same_dims_metrics,
)

DERIVATION_DIM = "derivation"
TARGET_COORD = "target"
PREDICT_COORD = "predict"


@pytest.fixture()
def ds_mock():
    da = xr.DataArray(
        [[[0.0, 1.0, 1.0]]],
        dims=["z", "x", DERIVATION_DIM],
        coords={
            "z": [0],
            "x": [0],
            DERIVATION_DIM: [TARGET_COORD, PREDICT_COORD, "mean"],
        },
    )
    delp = xr.DataArray([[1.0]], dims=["z", "x"], coords={"z": [0], "x": [0]})
    return xr.Dataset(
        {
            "dQ1": da,
            "dQ2": da,
            "column_integrated_dQ1": da,
            "column_integrated_dQ2": da,
            "Q1": da,
            "Q2": da,
            "column_integrated_Q1": da,
            "column_integrated_Q2": da,
            "pressure_thickness_of_atmospheric_layer": delp,
        }
    )


@pytest.fixture()
def area():
    return xr.DataArray([1], dims=["x"], coords={"x": [0]}).rename("area")


def test__calc_same_dims_metrics(ds_mock, area):
    ds = xr.merge([area, ds_mock])
    ds["area_weights"] = area / (area.mean())
    batch_metrics = _calc_same_dims_metrics(
        ds,
        dim_tag="scalar",
        vars=["column_integrated_dQ1", "column_integrated_dQ2"],
        weight_vars=["area_weights"],
        mean_dim_vars=None,
    )
    for var in list(batch_metrics.data_vars):
        assert isinstance(batch_metrics[var].values.item(), float)


@pytest.fixture()
def ds_calc_test():
    area_weights = xr.DataArray(
        [[0.5, 1.5]], dims=["y", "x"], coords={"x": [0, 1], "y": [0]}
    )
    delp_weights = xr.DataArray(
        [[[2.0, 0.0], [1.6, 0.4]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )
    da_target = xr.DataArray(
        [[[10.0, 20.0], [-10.0, -20.0]]],
        dims=["y", "x", "z"],
        coords={"z": [0, 1], "x": [0, 1], "y": [0]},
    )

    return xr.Dataset(
        {
            "area_weights": area_weights,
            "delp_weights": delp_weights,
            "target": da_target,
        }
    )


@pytest.mark.parametrize(
    "weight_vars, mean_dims, expected",
    (
        [["area_weights"], None, np.mean([5, 10, -15, -30.0])],
        [
            ["delp_weights", "area_weights"],
            None,
            np.mean([10, 0, -15 * 1.6, -30.0 * 0.4]),
        ],
        [["area_weights"], ["x", "y"], [-5, -10]],
    ),
)
def test__weighted_average(ds_calc_test, weight_vars, mean_dims, expected):
    weights = [ds_calc_test[weight_var] for weight_var in weight_vars]
    assert np.allclose(
        _weighted_average(ds_calc_test["target"], weights, mean_dims).values, expected
    )

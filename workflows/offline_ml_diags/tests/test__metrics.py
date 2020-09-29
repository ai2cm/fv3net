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


area_weights = xr.DataArray([0.5, 1.5], dims=["x"], coords={"x": [0, 1]})
delp_weights = xr.DataArray(
    [[2.0, 0.0], [1.6, 0.4]], dims=["x", "z"], coords={"z": [0, 1], "x": [0, 1]},
)


@pytest.fixture()
def ds_mock():
    da = xr.DataArray(
        [[[10.0, 20.0], [-10.0, -20.0]], [[-11.0, -21.0], [-11.0, -21.0]]],
        dims=[DERIVATION_DIM, "x", "z"],
        coords={
            DERIVATION_DIM: [TARGET_COORD, PREDICT_COORD],
            "x": [0, 1],
            "z": [0, 1],
        },
    )
    delp = xr.DataArray(
        [[2.0, 0.0], [1.6, 0.4]], dims=["x", "z"], coords={"z": [0, 1], "x": [0, 1]}
    )
    return xr.Dataset(
        {
            "dQ1": da,
            "dQ2": da,
            "column_integrated_dQ1": da.isel(z=0),
            "column_integrated_dQ2": da.isel(z=0),
            "Q1": da
            + np.random.uniform(10.0),  # offset by pQ, which is const in derivation dim
            "Q2": da + np.random.uniform(10.0),
            "column_integrated_Q1": da.isel(z=0) + np.random.uniform(10.0),
            "column_integrated_Q2": da.isel(z=0) + np.random.uniform(10.0),
            "pressure_thickness_of_atmospheric_layer": delp,
        }
    )


def test__calc_same_dims_metrics(ds_mock):
    batch_metrics = _calc_same_dims_metrics(
        ds_mock,
        dim_tag="scalar",
        vars=["column_integrated_dQ1", "column_integrated_dQ2"],
        weights=[area_weights],
        mean_dim_vars=None,
    )
    for var in list(batch_metrics.data_vars):
        assert isinstance(batch_metrics[var].values.item(), float)


@pytest.mark.parametrize(
    "weights, mean_dims, expected",
    (
        pytest.param(
            [area_weights],
            None,
            np.array(np.mean([5, 10, -15, -30.0])),
            id="area_weight_full_avg",
        ),
        pytest.param(
            [delp_weights, area_weights],
            None,
            np.array(np.mean([10, 0, -15 * 1.6, -30.0 * 0.4])),
            id="area_delp_weight_full_avg",
        ),
        pytest.param(
            [area_weights], ["x"], np.array([-5, -10]), id="area_weight_2d_avg"
        ),
    ),
)
def test__weighted_average(ds_mock, weights, mean_dims, expected):
    da = ds_mock["dQ1"].sel(derivation="target")
    avg = _weighted_average(da, weights, mean_dims).values
    assert np.allclose(avg, expected)


@pytest.mark.parametrize(
    "dim_tag, vars, weights, mean_dim_vars",
    [
        pytest.param(
            "scalar",
            ["column_integrated_dQ1", "column_integrated_Q1"],
            [area_weights],
            None,
            id="area_weighted_avg_2d_var",
        ),
        pytest.param(
            "scalar",
            ["dQ1", "Q1"],
            [area_weights, delp_weights],
            None,
            id="area_delp_weighted_avg_3d_var",
        ),
        pytest.param(
            "pressure_level",
            ["dQ1", "Q1"],
            [area_weights],
            ["x"],
            id="area_delp_weighted_avg_3d_var",
        ),
    ],
)
def test_Q_dQ_have_same_bias(ds_mock, dim_tag, vars, weights, mean_dim_vars):
    # pQ is not predicted by the ML model, thus Q = pQ + dQ
    # should have the same bias as dQ
    batch_metrics = _calc_same_dims_metrics(
        ds_mock,
        dim_tag=dim_tag,
        vars=vars,
        weights=weights,
        mean_dim_vars=mean_dim_vars,
    )
    for comparison in ["predict_vs_target", "mean_vs_target"]:
        var_0 = f"{dim_tag}/bias/{vars[0]}/{comparison}"
        var_1 = f"{dim_tag}/bias/{vars[1]}/{comparison}"
        assert np.allclose(batch_metrics[var_0], batch_metrics[var_1])

import pytest
import xarray as xr
import numpy as np

from offline_ml_diags._helpers import (
    sample_outside_train_range,
    drop_temperature_humidity_tendencies_if_not_predicted,
    _tendency_in_predictions,
    get_variable_indices,
    _count_features_2d,
)


def data_array_with_feature_dim(feature_dim: int, has_time_dim=True):
    da = xr.DataArray(
        [
            [
                [[[1 for x in range(2)] for y in range(2)] for z in range(feature_dim)]
                for tile in range(2)
            ]
        ],
        dims=["time", "tile", "z", "y", "x"],
    )
    if not has_time_dim:
        da = da.isel(time=0).squeeze(drop=True)
    if feature_dim == 1:
        da = da.isel(z=0).squeeze(drop=True)
    return da


def test_get_variable_indices():
    ds = xr.Dataset(
        {
            "dim_1": data_array_with_feature_dim(1, has_time_dim=True),
            "dim_10": data_array_with_feature_dim(10, has_time_dim=True),
            "dim_10_no_time": data_array_with_feature_dim(10, has_time_dim=False),
        }
    )
    variable_indices = get_variable_indices(ds, ["dim_1", "dim_10", "dim_10_no_time"])
    assert variable_indices == {
        "dim_1": (0, 1),
        "dim_10": (1, 11),
        "dim_10_no_time": (11, 21),
    }


@pytest.mark.parametrize(
    "tendency, output_variables, expected",
    [
        ("Q1", ["dQ1", "dQ2"], True),
        ("dQ1", ["dQ1", "dQ2"], True),
        ("Q1", ["dQu", "dQv"], False),
    ],
)
def test__tendency_in_predictions(tendency, output_variables, expected):
    assert _tendency_in_predictions(tendency, output_variables) == expected


@pytest.mark.parametrize(
    "vars, output_variables, vars_after_drop",
    [
        (
            [
                "dQ1",
                "column_integrated_dQ1",
                "dQ2",
                "Q1",
                "Q2",
                "dQu",
                "column_integrated_dQu",
            ],
            ["dQ1", "dQu"],
            ["dQ1", "column_integrated_dQ1", "Q1", "dQu", "column_integrated_dQu"],
        ),
        (
            ["dQ1", "column_integrated_dQ1", "dQ2", "Q1", "Q2"],
            ["dQ1"],
            ["dQ1", "column_integrated_dQ1", "Q1"],
        ),
    ],
)
def test_drop_temperature_humidity_tendencies_if_not_predicted(
    vars, output_variables, vars_after_drop
):
    da = xr.DataArray(0.0, dims=["x"], coords={"x": range(2)})
    ds = xr.Dataset({var: da for var in vars})
    ds_after_drop = drop_temperature_humidity_tendencies_if_not_predicted(
        ds, output_variables
    )
    assert set(ds_after_drop.data_vars) == set(vars_after_drop)


@pytest.mark.parametrize(
    "train, all, n, expected_length, allowed_test_samples",
    [
        pytest.param(
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            2,
            2,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
            id="request_less_than_available",
        ),
        pytest.param(
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            10,
            3,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
            id="request_more_than_available",
        ),
        pytest.param(
            [],
            ["20150101.120000", "20150102.120000"],
            1,
            1,
            ["20150101.120000", "20150102.120000"],
            id="no-config-set",
        ),
        pytest.param(
            ["20160101.120000", "20160102.120000"],
            ["20160101.120000", "20160102.120000"],
            3,
            None,
            None,
            id="error-none-avail-outside-config-range",
        ),
    ],
)
def test_sample_outside_train_range(
    train, all, n, expected_length, allowed_test_samples
):
    if expected_length:
        test = sample_outside_train_range(all, train, n)
        assert len(test) == expected_length
        assert set(test).issubset(allowed_test_samples)
    else:
        with pytest.raises(ValueError):
            test = sample_outside_train_range(all, train, n)


def test_count_features_2d():
    SAMPLE_DIM_NAME = "axy"
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.zeros([10]), dims=[SAMPLE_DIM_NAME]),
            "b": xr.DataArray(np.zeros([10, 1]), dims=[SAMPLE_DIM_NAME, "b_dim"]),
            "c": xr.DataArray(np.zeros([10, 5]), dims=[SAMPLE_DIM_NAME, "c_dim"]),
        }
    )
    names = list(ds.data_vars.keys())
    assert len(names) == 3
    out = _count_features_2d(names, ds, sample_dim_name=SAMPLE_DIM_NAME)
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5

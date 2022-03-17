import xarray as xr
import vcm
from fv3net.diagnostics.offline._helpers import (
    DATASET_DIM_NAME,
    compute_r2,
    insert_aggregate_bias,
    insert_aggregate_r2,
    insert_column_integrated_vars,
    rename_via_replace,
)


def test_compute_r2():
    ds = xr.Dataset(
        {
            "a_mse": xr.DataArray(1.0),
            "a_variance": xr.DataArray(2.0),
            "b": xr.DataArray(0),
        }
    )
    result = compute_r2(ds)
    expected = xr.Dataset({"a_r2": 0.5})
    xr.testing.assert_identical(result, expected)


def test_rename_via_replace():
    ds = xr.Dataset({"a_mse": xr.DataArray(0), "b_variance": xr.DataArray(0)})
    result = rename_via_replace(ds, "_mse", "_test")
    expected = xr.Dataset({"a_test": xr.DataArray(0), "b_variance": xr.DataArray(0)})
    xr.testing.assert_identical(result, expected)


def test_insert_aggregate_r2():
    ds = xr.Dataset(
        {
            "a_mse": xr.DataArray([0.5, 1.0], dims=[DATASET_DIM_NAME]),
            "a_variance": xr.DataArray([1.0, 4.0], dims=[DATASET_DIM_NAME]),
            "a_r2": xr.DataArray([0.5, 0.75], dims=[DATASET_DIM_NAME]),
            "b": xr.DataArray(0),
        }
    )
    result = insert_aggregate_r2(ds)
    expected = xr.Dataset(
        {
            "a_per_dataset_r2": xr.DataArray([0.5, 0.75], dims=[DATASET_DIM_NAME]),
            "a_r2": xr.DataArray(0.7),
            "a_mse": xr.DataArray([0.5, 1.0], dims=[DATASET_DIM_NAME]),
            "a_variance": xr.DataArray([1.0, 4.0], dims=[DATASET_DIM_NAME]),
            "b": xr.DataArray(0),
        }
    )
    xr.testing.assert_identical(result, expected)


def test_insert_aggregate_bias():
    ds = xr.Dataset(
        {
            "a_bias": xr.DataArray([1.0, 1.0], dims=[DATASET_DIM_NAME]),
            "b": xr.DataArray(0),
        }
    )
    result = insert_aggregate_bias(ds)
    expected = xr.Dataset(
        {
            "a_bias": xr.DataArray(1.0),
            "a_per_dataset_bias": xr.DataArray([1.0, 1.0], dims=[DATASET_DIM_NAME]),
            "b": xr.DataArray(0),
        }
    )
    xr.testing.assert_identical(result, expected)


def test_insert_column_integrated_vars():
    ds = xr.Dataset(
        {
            "Q1": xr.DataArray([1.0, 3.0], [("z", [0.0, 1.0])], ["z"]),
            "pressure_thickness_of_atmospheric_layer": xr.DataArray(
                [1.0, 1.0], [("z", [0.0, 1.0])], ["z"]
            ),
        }
    )

    heating = vcm.column_integrated_heating_from_isochoric_transition(
        ds["Q1"], ds["pressure_thickness_of_atmospheric_layer"]
    )
    expected = ds.assign({"column_integrated_Q1": heating})

    xr.testing.assert_allclose(insert_column_integrated_vars(ds, ["Q1"]), expected)

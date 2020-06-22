import pytest
import xarray as xr
import numpy as np
import diagnostics_utils as utils
from diagnostics_utils.utils import insert_column_integrated_vars, _rechunk_time_z
from vcm import thermo


def test_version():
    assert utils.__version__ == "0.1.0"


def test__inserted_column_integrated_vars():

    ds = xr.Dataset(
        {
            "Q1": xr.DataArray([1.0, 3.0], [("z", [0.0, 1.0])], ["z"]),
            "pressure_thickness_of_atmospheric_layer": xr.DataArray(
                [1.0, 1.0], [("z", [0.0, 1.0])], ["z"]
            ),
        }
    )

    expected = ds.assign(
        {
            "column_integrated_Q1": thermo.column_integrated_heating(
                ds["Q1"], ds["pressure_thickness_of_atmospheric_layer"]
            )
        }
    )

    xr.testing.assert_allclose(insert_column_integrated_vars(ds, ["Q1"]), expected)


def test__rechunk_time_z():

    da = xr.DataArray(
        np.ones((2, 4, 2)),
        [("time", [0.0, 1.0]), ("z", [0.0, 1.0, 2.0, 3.0]), ("x", [0.0, 1.0])],
        ["time", "z", "x"],
    )
    da_chunked_in_time = da.chunk({"time": 1, "z": 4, "x": 2})
    expected = da.chunk({"time": 2, "z": 1, "x": 2})

    assert expected.chunks == _rechunk_time_z(da_chunked_in_time).chunks


da = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])
da_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])
ds = xr.Dataset({"a": da})
weights = xr.DataArray([0.5, 0.5, 1, 1], dims=["z"])
weights_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])


@pytest.mark.parametrize(
    "da,weights,dims,expected",
    [
        (da, weights, "z", xr.DataArray(17.0 / 6.0)),
        (ds, weights, "z", xr.Dataset({"a": xr.DataArray(17.0 / 6.0)})),
        (da_nans, weights, "z", xr.DataArray(0.0)),
        (da, weights_nans, "z", xr.DataArray(np.nan)),
    ],
)
def test_weighted_average(da, weights, dims, expected):
    xr.testing.assert_allclose(utils.weighted_average(da, weights, dims), expected)


def test_weighted_averaged_no_dims():

    da = xr.DataArray([[[np.arange(1.0, 5.0)]]], dims=["tile", "y", "x", "z"])
    weights = xr.DataArray([[[[0.5, 0.5, 1, 1]]]], dims=["tile", "y", "x", "z"])
    expected = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])

    xr.testing.assert_allclose(utils.weighted_average(da, weights), expected)


enumeration = {"land": 1, "sea": 0}


@pytest.mark.parametrize(
    "float_mask,enumeration,atol,expected",
    [
        (
            xr.DataArray([1.0, 0.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray(["land", "sea"], dims=["x"]),
        ),
        (
            xr.DataArray([1.0000001, 0.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray(["land", "sea"], dims=["x"]),
        ),
        (
            xr.DataArray([1.0001, 0.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray([np.nan, "sea"], dims=["x"]),
        ),
    ],
)
def test_snap_mask_to_type(float_mask, enumeration, atol, expected):
    xr.testing.assert_equal(
        utils.snap_mask_to_type(float_mask, enumeration, atol), expected
    )


ds = xr.Dataset(
    {"a": xr.DataArray([[[np.arange(1.0, 5.0)]]], dims=["z", "tile", "y", "x"])}
)
surface_type_da = xr.DataArray(
    [[[["sea", "land", "land", "land"]]]], dims=["z", "tile", "y", "x"]
)
area = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["x"])


@pytest.mark.parametrize(
    "ds,surface_type_da,surface_type,area,expected",
    [
        (
            ds,
            surface_type_da,
            "sea",
            area,
            xr.Dataset({"a": xr.DataArray([1.0], dims=["z"])}),
        ),
        (
            ds,
            surface_type_da,
            "land",
            area,
            xr.Dataset({"a": xr.DataArray([3.0], dims=["z"])}),
        ),
    ],
)
def test__conditional_average(ds, surface_type_da, surface_type, area, expected):

    average = utils.conditional_average(ds, surface_type_da, surface_type, area)
    xr.testing.assert_allclose(average, expected)

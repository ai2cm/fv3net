import pytest
import xarray as xr
import numpy as np
import cftime
from numpy.random import randint
import numpy as np

from vcm.fv3.metadata import (
    _adjust_tile_range,
    _rename_dims,
    _set_missing_attrs,
    standardize_fv3_diagnostics,
    gfdl_to_standard,
    standard_to_gfdl
)


@pytest.mark.parametrize(
    "ds",
    [
        xr.Dataset(coords={"tile": np.arange(1, 7)}),
        xr.Dataset(coords={"tile": np.arange(6)}),
    ],
)
def test__check_tile_range(ds):

    expected = np.arange(6)
    expected_da = xr.DataArray(expected, dims=["tile"], coords={"tile": expected})
    tile_result = _adjust_tile_range(ds).tile
    xr.testing.assert_equal(tile_result, expected_da)


def _create_dataset(*dims, with_coords=True):
    if with_coords:
        coords = {dim: np.arange(i + 1) for i, dim in enumerate(dims)}
        ds = xr.Dataset(coords=coords)
    else:
        arr = np.zeros([i + 1 for i in range(len(dims))])
        da = xr.DataArray(arr, dims=dims)
        ds = xr.Dataset({"varname": da})
    return ds


@pytest.mark.parametrize(
    "input_dims, rename_inverse, renamed_dims",
    [
        ({"x", "y"}, {}, {"x", "y"}),
        ({"x", "y"}, {"y_out": {"y"}}, {"x", "y_out"}),
        ({"x", "y"}, {"y_out": {"y", "y2"}}, {"x", "y_out"}),
        ({"x", "y"}, {"x_out": {"x"}, "y_out": {"y", "y2"}}, {"x_out", "y_out"}),
        ({"x", "y"}, {"z_out": {"z"}}, {"x", "y"}),
    ],
)
def test__rename_dims(input_dims, rename_inverse, renamed_dims):
    # datasets can have dimensions with or without coordinates, so cover both cases
    for with_coords in [True, False]:
        ds_in = _create_dataset(*input_dims, with_coords=with_coords)
        ds_out = _rename_dims(ds_in, rename_inverse=rename_inverse)
        assert set(ds_out.dims) == renamed_dims


@pytest.fixture
def xr_darray():
    data = np.arange(16).reshape(4, 4)
    x = np.arange(4)
    y = np.arange(4)

    da = xr.DataArray(data, coords={"x": x, "y": y}, dims=["x", "y"],)

    return da


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"units": "best units"},
        {"long_name": "name is long!"},
        {"units": "trees", "long_name": "number of U.S. trees"},
    ],
)
def test__set_missing_attrs(attrs, xr_darray):

    xr_darray.attrs.update(attrs)
    res = _set_missing_attrs(xr_darray.to_dataset(name="data"))
    assert "long_name" in res.data.attrs
    assert "units" in res.data.attrs


def test__set_missing_attrs_description(xr_darray):

    attrs = {"description": "a description will be converted to a longname"}
    xr_darray.attrs.update(attrs)
    res = _set_missing_attrs(xr_darray.to_dataset(name="data"))
    assert res.data.attrs["long_name"] == attrs["description"]


DIM_NAMES = ["grid_xt", "grid_yt", "tile"]
DATA_VARS = ["a", "b"]
TIME_DIM = 10


def _create_raw_dataset():
    coords = {dim: np.arange(i + 1) for i, dim in enumerate(DIM_NAMES)}
    sizes = {dim: len(coords[dim]) for dim in DIM_NAMES}
    dataset = {}
    for data_var in DATA_VARS:
        arr = np.ones([*sizes.values()])
        dataset[data_var] = xr.DataArray(arr, dims=DIM_NAMES)
    ds = xr.Dataset(dataset, coords=coords)
    time_coord = [
        cftime.DatetimeJulian(2016, 1, n + 1, 0, 0, 0, randint(100))
        for n in range(TIME_DIM)
    ]
    ds["time"] = time_coord
    return ds


def test_standardize_fv3_diagnostics(tmpdir, regtest):
    ds = _create_raw_dataset()
    ds.to_zarr(str(tmpdir.join("fv3_diag.zarr")), consolidated=True)
    diag = xr.open_zarr(str(tmpdir.join("fv3_diag.zarr")), consolidated=True).load()
    standardized_ds = standardize_fv3_diagnostics(diag)
    with regtest:
        print(standardized_ds)
        print(standardized_ds.time)
        print(standardized_ds.time.attrs)


def test_gfdl_to_standard_dims_correct():
    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], np.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], np.ones((1, 1, 1, 1))),
        }
    )

    ans = gfdl_to_standard(data)
    assert set(ans.dims) == {"tile", "x", "y", "z"}


def test_gfdl_to_standard_is_inverse_of_standard_to_gfdl():

    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], np.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], np.ones((1, 1, 1, 1))),
        }
    )

    back = standard_to_gfdl(gfdl_to_standard(data))
    xr.testing.assert_equal(data, back)

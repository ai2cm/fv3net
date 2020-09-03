import pytest
import xarray as xr
import numpy as np

import load_diagnostic_data as load_diags


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
    tile_result = load_diags._adjust_tile_range(ds).tile
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
        ds_out = load_diags._rename_dims(ds_in, rename_inverse=rename_inverse)
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
    res = load_diags._set_missing_attrs(xr_darray.to_dataset(name="data"))
    assert "long_name" in res.data.attrs
    assert "units" in res.data.attrs


def test__set_missing_attrs_description(xr_darray):

    attrs = {"description": "a description will be converted to a longname"}
    xr_darray.attrs.update(attrs)
    res = load_diags._set_missing_attrs(xr_darray.to_dataset(name="data"))
    assert res.data.attrs["long_name"] == attrs["description"]


def test_warn_on_overwrite_duplicates():

    old = {"key"}
    new = ["duplicate", "duplicate"]
    with pytest.warns(UserWarning):
        load_diags.warn_on_overwrite(old, new)


def test_warn_on_overwrite_overlap():

    old = {"key"}
    new = ["key", "duplicate"]
    with pytest.warns(UserWarning):
        load_diags.warn_on_overwrite(old, new)


def test_warn_on_overwrite_no_warning():

    old = {"key"}
    new = {"new_key"}

    with pytest.warns(None) as records:
        load_diags.warn_on_overwrite(old, new)

    # ignore deprecation warnings
    for warning in records:
        assert not isinstance(warning, UserWarning), warning


def test__coarsen_keeps_attrs(xr_darray):
    ds = xr.Dataset({"var1": xr_darray, "var2": xr_darray})
    ds.attrs = {"global_attr": "value"}
    ds.var1.attrs = {"units": "value"}
    output = load_diags._coarsen(ds, xr_darray, 2)
    assert ds.attrs == output.attrs
    assert ds.var1.attrs == output.var1.attrs
    assert ds.var2.attrs == output.var2.attrs


def test__get_coarsening_args(xr_darray):
    target_res = 2
    grid_entries = {4: "c4_grid_entry"}
    grid, coarsening_factor = load_diags._get_coarsening_args(
        xr_darray, target_res, grid_entries=grid_entries
    )
    assert grid == "c4_grid_entry"
    assert coarsening_factor == 2


@pytest.mark.parametrize(
    "input_dims", [("x", "time"), ("x", "y")],
)
def test__set_calendar_to_julian(input_dims):
    ds = _create_dataset(*input_dims, with_coords=True)
    ds_out = load_diags._set_calendar_to_julian(ds)
    if "time" in input_dims:
        assert ds_out.time.attrs["calendar"] == "julian"

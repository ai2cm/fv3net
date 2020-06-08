import pytest
import itertools
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


def _create_ds_from_dims(*dims):

    coords = {dim: np.arange(i + 1) for i, dim in enumerate(dims)}
    return xr.Dataset(coords=coords)


def _check_keys_equivalent(target_keys, test_keys):

    assert len(test_keys) == len(target_keys)
    for key in target_keys:
        assert key in test_keys


def test__rename_coords():
    """
    Construct DS from combos of source dimension names and check that it gets
    fixed with renaming function
    """

    good_coord_keys = load_diags.COORD_RENAME_INVERSE_MAP.keys()
    good_ds = _create_ds_from_dims(*good_coord_keys)
    good_ds_renamed = load_diags._rename_coords(good_ds)
    _check_keys_equivalent(good_coord_keys, good_ds_renamed.coords.keys())

    for bad_dims in itertools.product(*load_diags.COORD_RENAME_INVERSE_MAP.values()):
        ds = _create_ds_from_dims(*bad_dims)
        fixed_ds = load_diags._rename_coords(ds)
        fixed_coord_keys = fixed_ds.coords.keys()
        _check_keys_equivalent(good_coord_keys, fixed_coord_keys)


@pytest.fixture
def xr_darray():
    data = np.arange(20).reshape(4, 5)
    x = np.arange(4)
    y = np.arange(5)

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

    with pytest.warns(None) as record:
        load_diags.warn_on_overwrite(old, new)

    assert len(record) == 0

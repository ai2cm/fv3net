import pytest
import itertools
import xarray as xr
import numpy as np

import load_diagnostic_data as load_diags

@pytest.mark.parametrize("ds",
    [
        xr.Dataset(coords={"tile": np.arange(1, 7)}),
        xr.Dataset(coords={"tile": np.arange(6)}),
    ]
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

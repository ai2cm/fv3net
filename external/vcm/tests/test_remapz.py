import numpy as np
import pytest
import xarray as xr
from vcm.cubedsphere.remapz import remap_levels, _mask_weights


def input_dataarray(shape):
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    # only use last len(shape) dimension names
    dims = ["t", "x", "y", "z"]
    dims = dims[len(dims) - len(shape) :]
    return xr.DataArray(data, dims=dims, coords=None, name="foo")


@pytest.mark.parametrize(
    "p_in_shape, f_in_shape, p_out_shape, expected",
    [
        ((4, 6), (4, 5), (4, 6), (4, 5)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 6), (4, 4, 5)),
        ((6, 4, 4, 6), (6, 4, 4, 5), (6, 4, 4, 6), (6, 4, 4, 5)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 3), (4, 4, 2)),
    ],
)
def test_remap_levels(p_in_shape, f_in_shape, p_out_shape, expected):
    p_in = input_dataarray(p_in_shape)
    f_in = input_dataarray(f_in_shape)
    p_out = input_dataarray(p_out_shape)
    try:
        f_out = remap_levels(p_in, f_in, p_out, z_dim_center="z", z_dim_outer="z")
    except ImportError:
        pytest.skip("mappm import failed. Skipping test_remap_levels.")
    f_out = f_out.transpose(*f_in.dims)
    assert f_out.shape == expected


def test__mask_weights():
    weights = input_dataarray((2, 2, 1)).isel(z=0)
    phalf_coarse_on_fine = input_dataarray((2, 2, 3))
    phalf_fine = input_dataarray((2, 2, 3))
    phalf_coarse_on_fine[:, :, 0] = 1.0
    phalf_coarse_on_fine[:, :, 1] = 2.0
    phalf_coarse_on_fine[:, :, 2] = 3.0
    phalf_fine[:, :, 0] = 1.0
    phalf_fine[:, :, 1] = 2.0
    phalf_fine[0, :, 2] = 2.5
    phalf_fine[1, :, 2] = 3.5
    expected_weights = weights.broadcast_like(phalf_fine.isel(z=slice(None, -1))).copy()
    expected_weights[0, :, 1] = 0.0
    masked_weights = _mask_weights(
        weights, phalf_coarse_on_fine, phalf_fine, dim_center="z", dim_outer="z"
    )
    xr.testing.assert_allclose(expected_weights, masked_weights)

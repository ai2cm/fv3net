import numpy as np
import pytest
import xarray as xr
from vcm.cubedsphere.remapz import (
        remap_levels,
    )
try:
    import mappm
except:
    mappm_import_failed = True
else:
    mappm_import_failed = False
from vcm.cubedsphere.constants import RESTART_Z_CENTER


def input_dataarray(shape):
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    dims = ["x", "y", "z"]
    return xr.DataArray(data, dims=dims, coords=None, name="foo")


@pytest.mark.parametrize(
    "p_in_shape, f_in_shape, p_out_shape, expected",
    [
        ((4, 4, 6), (4, 4, 5), (4, 4, 6), (4, 4, 5)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 3), (4, 4, 2)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 6), (4, 4, 5)),
    ],
)
def test_remap_levels(p_in_shape, f_in_shape, p_out_shape, expected):
    if mappm_import_failed:
        pytest.skip("mappm import failed. Skipping test_remap_levels.")
    p_in = input_dataarray(p_in_shape)
    f_in = input_dataarray(f_in_shape)
    p_out = input_dataarray(p_out_shape)
    f_out = remap_levels(p_in, f_in, p_out, dim="z")
    assert f_out.shape == expected.shape

import dask
import numpy as np
import pytest
import xarray as xr
from vcm.cubedsphere.regridz import (
    regrid_vertical_legacy,
    _mask_weights,
    regrid_vertical,
)


try:
    import mappm  # noqa: F401
except ImportError:
    has_mappm = False
else:
    has_mappm = True


def input_dataarray(shape, chunks=None, z_dim_name="z"):
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    # only use last len(shape) dimension names
    dims = ["t", "x", "y", z_dim_name]
    dims = dims[len(dims) - len(shape) :]
    da = xr.DataArray(data, dims=dims, coords=None, name="foo")
    if chunks is not None:
        da = da.chunk({dim: i for dim, i in zip(dims, chunks)})
    return da


@pytest.mark.skipif(not has_mappm, reason="test requires mappm")
@pytest.mark.parametrize(
    "p_in_shape, f_in_shape, p_out_shape, expected",
    [
        ((4, 6), (4, 5), (4, 6), (4, 5)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 6), (4, 4, 5)),
        ((6, 4, 4, 6), (6, 4, 4, 5), (6, 4, 4, 6), (6, 4, 4, 5)),
        ((4, 4, 6), (4, 4, 5), (4, 4, 3), (4, 4, 2)),
    ],
)
def test_regrid_vertical_against_regrid_vertical_legacy(
    p_in_shape, f_in_shape, p_out_shape, expected
):
    p_in = input_dataarray(p_in_shape, z_dim_name="z_outer")
    f_in = input_dataarray(f_in_shape, z_dim_name="z_center")
    p_out = input_dataarray(p_out_shape, z_dim_name="z_outer")
    f_out_legacy = regrid_vertical_legacy(
        p_in, f_in, p_out, z_dim_center="z_center", z_dim_outer="z_outer"
    )
    f_out_current = regrid_vertical(
        p_in, f_in, p_out, z_dim_center="z_center", z_dim_outer="z_outer"
    )

    # The legacy version of regrid_vertical adds coordinates where they don't
    # exist; therefore we'll only check for equality of the arrays themselves here.
    np.testing.assert_array_equal(f_out_legacy, f_out_current)


@pytest.mark.skipif(not has_mappm, reason="test requires mappm")
@pytest.mark.parametrize(
    [
        "p_in_shape",
        "p_in_chunks",
        "f_in_shape",
        "f_in_chunks",
        "p_out_shape",
        "p_out_chunks",
        "expected_shape",
    ],
    [
        ((4, 6), (1, -1), (4, 5), (1, -1), (4, 6), (1, -1), (4, 5)),
        ((4, 6), (1, 1), (4, 5), (2, -1), (4, 6), (3, 1), (4, 5)),
        ((4, 6), (1, 1), (4, 5), None, (4, 6), (3, 1), (4, 5)),
        ((4, 4, 6), (1, 1, -1), (4, 4, 5), (2, 3, -1), (4, 4, 6), (3, 2, 1), (4, 4, 5)),
        ((4, 4, 6), (1, 1, -1), (4, 4, 5), (2, 3, -1), (4, 4, 3), (3, 2, 1), (4, 4, 2)),
    ],
    ids=[
        "2d-contiguous-chunks",
        "2d-non-contiguous-chunks",
        "2d-one-unchunked-array",
        "3d-non-contiguous-chunks",
        "3d-non-contiguous-chunks-new-nlevels",
    ],
)
def test_regrid_vertical_dask(
    p_in_shape,
    p_in_chunks,
    f_in_shape,
    f_in_chunks,
    p_out_shape,
    p_out_chunks,
    expected_shape,
):
    p_in = input_dataarray(p_in_shape, chunks=p_in_chunks, z_dim_name="z_outer")
    f_in = input_dataarray(f_in_shape, chunks=f_in_chunks, z_dim_name="z_center")
    p_out = input_dataarray(p_out_shape, chunks=p_out_chunks, z_dim_name="z_outer")

    f_out = regrid_vertical(
        p_in, f_in, p_out, z_dim_center="z_center", z_dim_outer="z_outer"
    )
    assert isinstance(f_out.data, dask.array.Array)
    f_out_numpy = regrid_vertical(
        p_in.compute(),
        f_in.compute(),
        p_out.compute(),
        z_dim_center="z_center",
        z_dim_outer="z_outer",
    )
    xr.testing.assert_identical(f_out.compute(), f_out_numpy)


@pytest.mark.skipif(not has_mappm, reason="test requires mappm")
def test_regrid_vertical_invalid_dimension_names():
    p_in = input_dataarray((4, 6), z_dim_name="z")
    f_in = input_dataarray((4, 5), z_dim_name="z")
    p_out = input_dataarray((4, 3), z_dim_name="z")
    with pytest.raises(ValueError, match="must not be equal"):
        regrid_vertical(p_in, f_in, p_out, z_dim_center="z", z_dim_outer="z")


@pytest.mark.skipif(not has_mappm, reason="test requires mappm")
def test_regrid_vertical_invalid_columns():
    p_in = input_dataarray((4, 6), z_dim_name="z_outer")
    f_in = input_dataarray((3, 5), z_dim_name="z_center")
    p_out = input_dataarray((4, 3), z_dim_name="z_outer")
    with pytest.raises(ValueError, match="must be same size"):
        regrid_vertical(
            p_in, f_in, p_out, z_dim_center="z_center", z_dim_outer="z_outer"
        )


@pytest.mark.skipif(not has_mappm, reason="test requires mappm")
def test_regrid_vertical_invalid_vertical_dimension_size():
    p_in = input_dataarray((4, 6), z_dim_name="z_outer")
    f_in = input_dataarray((4, 3), z_dim_name="z_center")
    p_out = input_dataarray((4, 3), z_dim_name="z_outer")
    with pytest.raises(ValueError, match="one shorter than p_in"):
        regrid_vertical(
            p_in, f_in, p_out, z_dim_center="z_center", z_dim_outer="z_outer"
        )


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

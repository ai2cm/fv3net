import numpy as np
import pytest
import xarray as xr

from fv3fit.reservoir.cubed_sphere import CubedSphereDivider


NTILE = 6


def cubed_sphere_data(nx, ny, nz=2, nt=5):
    tile_data = np.arange(nx * ny * nz * nt).reshape(nx, ny, nz, nt)
    da = xr.DataArray(
        data=np.array([100 * t + tile_data for t in range(1, NTILE + 1)]),
        dims=["tile", "x", "y", "z", "time"],
    )
    return xr.Dataset({"var_3d": da, "var_2d": da.isel(z=0)})


def test_CubedSphereDivider_get_rank_data_values():
    data = cubed_sphere_data(nx=4, ny=4, nt=1, nz=1).isel({"time": 0, "z": 0})
    divider = CubedSphereDivider(
        tile_layout=(2, 2),
        global_dims=list(data.dims),
        global_extent=[data[dim].size for dim in data.dims],
    )
    # corner at x=0, y=0 on first tile
    rank0_data = divider.get_rank_data(data, rank=0, overlap=0)
    np.testing.assert_array_almost_equal(
        rank0_data["var_2d"].values, np.array([[100, 101], [104, 105]])
    )

    # shifts along x axis (axis 0) on first tile
    rank1_data = divider.get_rank_data(data, rank=1, overlap=0)
    np.testing.assert_array_almost_equal(
        rank1_data["var_2d"].values, np.array([[108, 109], [112, 113]])
    )

    # corner at x=0, y=0 on 2nd tile
    rank5_data = divider.get_rank_data(data, rank=4, overlap=0)
    np.testing.assert_array_almost_equal(
        rank5_data["var_2d"].values, np.array([[200, 201], [204, 205]])
    )


@pytest.mark.parametrize(
    "layout, overlap, expected_rank_shape",
    [
        [(2, 2), 0, (2, 2)],
        [(2, 2), 1, (4, 4)],
        [(1, 1), 0, (4, 4)],
        [(1, 1), 2, (8, 8)],
    ],
)
def test_CubedSphereDivider_get_rank_data(layout, overlap, expected_rank_shape):
    data = cubed_sphere_data(nx=4, ny=4)
    divider = CubedSphereDivider(
        tile_layout=layout,
        global_dims=list(data.dims),
        global_extent=[data[dim].size for dim in data.dims],
    )
    rank_data = divider.get_rank_data(data, rank=0, overlap=overlap)
    rank_xy_shape = (rank_data["x"].size, rank_data["y"].size)
    assert rank_xy_shape == expected_rank_shape


def test_CubedSphereDivider_get_rank_data_overlap():
    """
    It is hard to know the layout of neighboring ranks with calling the
    CubedSpherePartitioner to get the answer, which is part of the code tested.
    Instead of directly testing the values for halo points I am testing that they
    belong to other tiles (>=200), and that the corner is zero. The expected output:
        [100, 101, >=200],
        [102, 103, >=200],
        [>=200, >=200, 0]

    """
    data = cubed_sphere_data(nx=2, ny=2, nt=1, nz=1).isel({"time": 0, "z": 0})
    divider = CubedSphereDivider(
        tile_layout=(2, 2),
        global_dims=list(data.dims),
        global_extent=[data[dim].size for dim in data.dims],
    )
    # corner at x=1, y=1 on first tile
    rank_data = divider.get_rank_data(data, rank=3, overlap=1)["var_2d"].values

    np.testing.assert_array_almost_equal(
        rank_data[:-1, :-1], np.array([[100, 101], [102, 103]])
    )
    assert all(v >= 200 for v in rank_data[-1, :-1])
    assert all(v >= 200 for v in rank_data[:-1, -1])
    assert rank_data[-1, -1] == 0.0

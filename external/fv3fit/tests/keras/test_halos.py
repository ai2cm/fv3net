from fv3fit.keras._models.shared import append_halos
import xarray as xr
import numpy as np
import pytest


def get_dataset(nx: int, ny: int, nz: int, n_tile: int) -> xr.Dataset:
    nt = 5
    ds = xr.Dataset(
        data_vars={
            "scalar": xr.DataArray(
                data=np.ones([nt, n_tile, nx, ny]),
                dims=["sample", "tile", "x", "y"],
                attrs={"units": ""},
            ),
            "scalar_single_precision": xr.DataArray(
                data=np.ones([nt, n_tile, nx, ny], dtype=np.float32),
                dims=["sample", "tile", "x", "y"],
                attrs={"units": ""},
            ),
            "column": xr.DataArray(
                data=np.ones([nt, n_tile, nx, ny, nz]),
                dims=["sample", "tile", "x", "y", "z"],
                attrs={"units": ""},
            ),
        }
    )
    return ds


@pytest.mark.parametrize(
    "nx, ny, nz, n_tile, n_halo",
    [
        pytest.param(8, 8, 12, 6, 3, id="typical"),
        pytest.param(8, 8, 12, 6, 1, id="one_halo"),
        pytest.param(8, 8, 12, 6, 0, id="no_halo"),
    ],
)
def test_append_halos_extends_dims(nx: int, ny: int, nz: int, n_tile: int, n_halo: int):
    ds = get_dataset(nx=nx, ny=ny, nz=nz, n_tile=n_tile)
    for name, da in ds.data_vars.items():
        assert da.sizes["x"] == nx, name
        assert da.sizes["y"] == ny, name
        assert da.sizes.get("z", nz) == nz, name
    result: xr.Dataset = append_halos(ds, n_halo=n_halo)
    for name, da in result.data_vars.items():
        assert da.sizes["x"] == nx + 2 * n_halo
        assert da.sizes["y"] == ny + 2 * n_halo, name
        assert da.sizes.get("z", nz) == nz, name
        # only corners should still be zero
        compute_data = da.isel(
            x=range(n_halo, n_halo + nx), y=range(n_halo, n_halo + ny)
        )
        # compute data was never zero, shouldn't be now
        assert np.sum(compute_data == 0) == 0
        # x and y edge halos should be 1 now, but corners are still zero
        data_with_x_halos = da.isel(
            x=range(0, nx + 2 * n_halo), y=range(n_halo, n_halo + ny)
        )
        assert np.sum(data_with_x_halos == 0) == 0
        data_with_y_halos = da.isel(
            x=range(n_halo, n_halo + nx), y=range(0, ny + 2 * n_halo)
        )
        assert np.sum(data_with_y_halos == 0) == 0
    assert result.dims.keys() == ds.dims.keys()


@pytest.mark.parametrize(
    "nx, ny, nz, n_tile, n_halo", [pytest.param(8, 8, 12, 1, 3, id="typical")]
)
def test_append_halos_raises_with_one_tile(
    nx: int, ny: int, nz: int, n_tile: int, n_halo: int
):
    ds = get_dataset(nx=nx, ny=ny, nz=nz, n_tile=n_tile)
    with pytest.raises(ValueError):
        append_halos(ds, n_halo=n_halo)


@pytest.mark.parametrize(
    "nx, ny, nz, n_tile", [pytest.param(8, 8, 12, 6, id="typical")]
)
def test_append_halos_no_halos_is_unchanged(nx: int, ny: int, nz: int, n_tile: int):
    ds = get_dataset(nx=nx, ny=ny, nz=nz, n_tile=n_tile)
    result: xr.Dataset = append_halos(ds, n_halo=0)
    xr.testing.assert_identical(result, ds)

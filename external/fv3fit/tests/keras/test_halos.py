from typing import Any, Dict, List
from fv3fit.keras._models.shared import append_halos
from fv3fit.keras._models.shared.halos import _append_halos_using_mpi
import xarray as xr
import numpy as np
import pytest
import pace.util


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
    check_result(result=result, n_halo=n_halo, nx=nx, ny=ny, nz=nz)
    assert result.dims.keys() == ds.dims.keys()


def check_result(result: xr.Dataset, n_halo: int, nx: int, ny: int, nz: int):
    for name, da in result.data_vars.items():
        assert da.sizes["x"] == nx + 2 * n_halo, name
        assert da.sizes["y"] == ny + 2 * n_halo, name
        assert da.sizes.get("z", nz) == nz, name
        # only corners should still be zero
        compute_data = da.isel(
            x=range(n_halo, n_halo + nx), y=range(n_halo, n_halo + ny)
        )
        # compute data was never zero, shouldn't be now
        assert np.sum(compute_data == 0) == 0, name
        # x and y edge halos should be 1 now, but corners are still zero
        data_with_x_halos = da.isel(
            x=range(0, nx + 2 * n_halo), y=range(n_halo, n_halo + ny)
        )
        assert np.sum(data_with_x_halos == 0) == 0, name
        data_with_y_halos = da.isel(
            x=range(n_halo, n_halo + nx), y=range(0, ny + 2 * n_halo)
        )
        assert np.sum(data_with_y_halos == 0) == 0, name
        # corner data should still be zeros
        for x_range in (range(0, n_halo), range(n_halo + nx, n_halo * 2 + nx)):
            for y_range in (range(0, n_halo), range(n_halo + ny, n_halo * 2 + ny)):
                corner_data = da.isel(x=x_range, y=y_range)
                assert np.all(corner_data.values == 0)


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


class ChainedDummyComm(pace.util.testing.DummyComm):
    """
    Dummy comm that calls a callback just before recv, allowing us to call all sends
    before recvs.

    Can be used to mock MPI behavior in a function that has a single send-recv pair.
    """

    def __init__(self, rank, total_ranks, buffer_dict):
        super().__init__(rank=rank, total_ranks=total_ranks, buffer_dict=buffer_dict)
        self._callback = lambda: None

    def set_callback(self, callback):
        self._callback = callback

    def Recv(self, *args, **kwargs):
        self._callback()
        self._callback = lambda: None
        super().Recv(*args, **kwargs)


def _get_callback(output_datasets, i: int, rank_dataset, n_halo, comm):
    def callback():
        output_datasets[i] = _append_halos_using_mpi(
            ds=rank_dataset, n_halo=n_halo, comm=comm
        )

    return callback


@pytest.mark.parametrize(
    "nx, ny, nz, n_halo",
    [pytest.param(8, 8, 12, 3, id="typical"), pytest.param(8, 8, 12, 1, id="one_halo")],
)
def test_with_and_without_mpi_give_same_result(nx: int, ny: int, nz: int, n_halo: int):
    n_tile = 6
    buffer_dict: Dict[Any, Any] = {}
    comms: List[ChainedDummyComm] = []
    for rank in range(n_tile):
        comms.append(
            ChainedDummyComm(rank=rank, total_ranks=n_tile, buffer_dict=buffer_dict)
        )
    full_ds = get_dataset(nx=nx, ny=ny, nz=nz, n_tile=n_tile).drop(
        ["scalar", "scalar_single_precision"]
    )
    rank_datasets = [full_ds.isel(tile=i) for i in range(n_tile)]
    non_mpi_result: xr.Dataset = append_halos(full_ds, n_halo=n_halo)

    output_datasets = [None for _ in range(n_tile)]
    for i in range(1, n_tile):
        comms[i - 1].set_callback(
            _get_callback(
                output_datasets,
                i=i,
                rank_dataset=rank_datasets[i],
                n_halo=n_halo,
                comm=comms[i],
            )
        )
    output_datasets[0] = _append_halos_using_mpi(
        ds=rank_datasets[0], n_halo=n_halo, comm=comms[0]
    )
    for i in range(n_tile):
        check_result(result=output_datasets[i], n_halo=n_halo, nx=nx, ny=ny, nz=nz)
        xr.testing.assert_identical(non_mpi_result.isel(tile=i), output_datasets[i])

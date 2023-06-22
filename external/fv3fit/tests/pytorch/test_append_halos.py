from fv3fit._shared.halos import append_halos
from fv3fit.pytorch.cyclegan.modules import AppendHalos
from fv3fit.pytorch.system import DEVICE
import xarray as xr
import numpy as np
import pytest
import torch


def get_dataset(nx: int, ny: int, nz: int, n_tile: int) -> xr.Dataset:
    nt = 5
    ds = xr.Dataset(
        data_vars={
            "data": xr.DataArray(
                data=np.random.randn(nt, n_tile, nz, nx, ny),
                dims=["sample", "tile", "z", "x", "y"],
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
def test_compare_against_reference_implementation(
    nx: int, ny: int, nz: int, n_tile: int, n_halo: int
):
    np.random.seed(0)
    ds = get_dataset(nx=nx, ny=ny, nz=nz, n_tile=n_tile)
    for name, da in ds.data_vars.items():
        assert da.sizes["x"] == nx, name
        assert da.sizes["y"] == ny, name
        assert da.sizes.get("z", nz) == nz, name
    append = AppendHalos(n_halo=n_halo)
    result = append(torch.as_tensor(ds.data.values).float().to(DEVICE))
    reference_ds: xr.Dataset = append_halos(ds, n_halo=n_halo)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(3, 6, figsize=(12, 4))
    # for i in range(6):
    #     ax[0, i].imshow(result[0, i, 0, :, :].cpu().numpy())
    #     ax[1, i].imshow(reference_ds.data[0, i, 0, :, :].values)
    #     ax[2, i].imshow(
    #         result[0, i, 0, :, :].cpu().numpy()
    #         - reference_ds.data[0, i, 0, :, :].values
    #     )
    # plt.show()

    np.testing.assert_almost_equal(result.cpu().numpy(), reference_ds.data.values)

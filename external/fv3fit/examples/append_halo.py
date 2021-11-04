import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from fv3fit.keras._models.shared import append_halos


if __name__ == "__main__":
    ds = xr.Dataset(
        data_vars={
            "data": xr.DataArray(
                np.random.randn(6, 12, 12, 1), dims=["tile", "x", "y", "z"],
            )
        }
    )

    plt.figure()
    plt.pcolormesh(ds["data"].values[0, :, :, 0], vmin=-3, vmax=3)
    plt.tight_layout()

    halo_ds = append_halos(ds, n_halo=3)

    plt.figure()
    plt.pcolormesh(halo_ds["data"].values[0, :, :, 0], vmin=-3, vmax=3)
    plt.tight_layout()
    plt.show()

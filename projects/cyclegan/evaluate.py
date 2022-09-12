import fv3fit
from matplotlib import pyplot as plt
import xarray as xr

if __name__ == "__main__":
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load("output").to("cpu")
    c48_real = (
        xr.open_zarr("c48_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=range(0, 100, 10))
    )
    c384_real = (
        xr.open_zarr("c384_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=range(0, 100, 10))
    )
    c48_gen = cyclegan.predict(c384_real)
    c384_gen = cyclegan.predict(c48_real, reverse=True)
    i_tile = 3
    for i_tile in range(1):
        for i in range(1):
            import pdb

            pdb.set_trace()
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            c48_real.h500.isel(time=i, tile=i_tile).plot(ax=ax[0, 0])
            ax[0, 0].set_title("c48_real")
            c384_real.h500.isel(time=i, tile=i_tile).plot(ax=ax[1, 0])
            ax[1, 0].set_title("c384_real")
            c384_gen.h500.isel(time=i, tile=i_tile).plot(ax=ax[0, 1])
            ax[0, 1].set_title("c384_gen")
            c48_gen.h500.isel(time=i, tile=i_tile).plot(ax=ax[1, 1])
            ax[1, 1].set_title("c48_gen")
            plt.tight_layout()
            plt.show()

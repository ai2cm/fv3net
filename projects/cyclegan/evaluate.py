# flake8: noqa

import random
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs


if __name__ == "__main__":
    random.seed(0)
    grid = catalog["grid/c48"].read()
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load("output_good").to(DEVICE)
    c48_real: xr.Dataset = (
        xr.open_zarr("c48_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(-2905 * 2, None, 2))
    ).load()
    c384_real: xr.Dataset = (
        xr.open_zarr("c384_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(-2905 * 2, None, 2))
    ).load()
    c384_gen: xr.Dataset = cyclegan.predict(c48_real)
    c48_gen: xr.Dataset = cyclegan.predict(c384_real, reverse=True)

    # for _ in range(3):
    #     i_time = random.randint(0, c48_real.time.size - 1)
    #     fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    #     fv3viz.plot_cube(ds=c48_real.isel(time=i_time).merge(grid), var_name="h500", ax=ax[0, 0])
    #     ax[0, 0].set_title("c48_real")
    #     fv3viz.plot_cube(ds=c384_real.isel(time=i_time).merge(grid), var_name="h500", ax=ax[1, 0])
    #     ax[1, 0].set_title("c384_real")
    #     fv3viz.plot_cube(ds=c384_gen.isel(time=i_time).merge(grid), var_name="h500", ax=ax[0, 1])
    #     ax[0, 1].set_title("c384_gen")
    #     fv3viz.plot_cube(ds=c48_gen.isel(time=i_time).merge(grid), var_name="h500", ax=ax[1, 1])
    #     ax[1, 1].set_title("c48_gen")
    #     plt.tight_layout()
    #     fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    #     fv3viz.plot_cube(ds=c48_real.isel(time=i_time).merge(grid), var_name="h500", ax=ax[0, 0])
    #     ax[0, 0].set_title("c48_real")
    #     fv3viz.plot_cube(ds=c384_real.isel(time=i_time).merge(grid), var_name="h500", ax=ax[1, 0])
    #     ax[1, 0].set_title("c384_real")
    #     fv3viz.plot_cube(ds=(c384_gen - c48_real).isel(time=i_time).merge(grid), var_name="h500", ax=ax[0, 1])
    #     ax[0, 1].set_title("c384_gen")
    #     fv3viz.plot_cube(ds=(c48_gen - c384_real).isel(time=i_time).merge(grid), var_name="h500", ax=ax[1, 1])
    #     ax[1, 1].set_title("c48_gen")
    #     plt.tight_layout()

    c48_real_mean = c48_real.mean("time")
    c48_gen_mean = c48_gen.mean("time")
    c384_real_mean = c384_real.mean("time")
    c384_gen_mean = c384_gen.mean("time")
    fig, ax = plt.subplots(
        2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    fv3viz.plot_cube(ds=c48_real_mean.merge(grid), var_name="h500", ax=ax[0, 0])
    ax[0, 0].set_title("c48_real")
    fv3viz.plot_cube(ds=c384_real_mean.merge(grid), var_name="h500", ax=ax[1, 0])
    ax[1, 0].set_title("c384_real")
    fv3viz.plot_cube(ds=c384_gen_mean.merge(grid), var_name="h500", ax=ax[0, 1])
    ax[0, 1].set_title("c384_gen")
    fv3viz.plot_cube(ds=c48_gen_mean.merge(grid), var_name="h500", ax=ax[1, 1])
    ax[1, 1].set_title("c48_gen")
    plt.tight_layout()

    fig, ax = plt.subplots(
        2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    fv3viz.plot_cube(ds=c48_real_mean.merge(grid), var_name="h500", ax=ax[0, 0])
    ax[0, 0].set_title("c48_real")
    fv3viz.plot_cube(
        ds=(c384_real_mean - c48_real_mean).merge(grid),
        var_name="h500",
        ax=ax[1, 0],
        vmin=-70,
        vmax=70,
    )
    ax[1, 0].set_title("c384_real")
    fv3viz.plot_cube(
        ds=(c384_gen_mean - c48_real_mean).merge(grid),
        var_name="h500",
        ax=ax[0, 1],
        vmin=-70,
        vmax=70,
    )
    ax[0, 1].set_title("c384_gen")
    fv3viz.plot_cube(
        ds=(c48_gen_mean - c48_real_mean).merge(grid),
        var_name="h500",
        ax=ax[1, 1],
        vmin=-70,
        vmax=70,
    )
    ax[1, 1].set_title("c48_gen")
    plt.tight_layout()

    mse = (c384_real_mean - c384_gen_mean).var()
    var = (c384_real_mean).var()
    print(mse)
    print(var)
    print(1.0 - (mse / var))

    mse = (c384_real_mean - c48_real_mean).var()
    var = (c384_real_mean).var()
    print(mse)
    print(var)
    print(1.0 - (mse / var))

    # c48_real_std = c48_real.std("time")
    # c48_gen_std = c48_gen.std("time")
    # c384_real_std = c384_real.std("time")
    # c384_gen_std = c384_gen.std("time")
    # fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    # fv3viz.plot_cube(ds=c48_real_std.merge(grid), var_name="h500", ax=ax[0, 0])
    # ax[0, 0].set_title("c48_real")
    # fv3viz.plot_cube(ds=c384_real_std.merge(grid), var_name="h500", ax=ax[1, 0])
    # ax[1, 0].set_title("c384_real")
    # fv3viz.plot_cube(ds=c384_gen_std.merge(grid), var_name="h500", ax=ax[0, 1])
    # ax[0, 1].set_title("c384_gen")
    # fv3viz.plot_cube(ds=c48_gen_std.merge(grid), var_name="h500", ax=ax[1, 1])
    # ax[1, 1].set_title("c48_gen")
    # plt.tight_layout()

    # fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    # fv3viz.plot_cube(ds=c48_real_std.merge(grid), var_name="h500", ax=ax[0, 0])
    # ax[0, 0].set_title("c48_real")
    # fv3viz.plot_cube(ds=(c384_real_std - c48_real_std).merge(grid), var_name="h500", ax=ax[1, 0], vmin=-50, vmax=50)
    # ax[1, 0].set_title("c384_real")
    # fv3viz.plot_cube(ds=(c384_gen_std - c48_real_std).merge(grid), var_name="h500", ax=ax[0, 1], vmin=-50, vmax=50)
    # ax[0, 1].set_title("c384_gen")
    # fv3viz.plot_cube(ds=(c48_gen_std - c48_real_std).merge(grid), var_name="h500", ax=ax[1, 1], vmin=-50, vmax=50)
    # ax[1, 1].set_title("c48_gen")
    # plt.tight_layout()

    plt.show()

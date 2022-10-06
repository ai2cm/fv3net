# flake8: noqa

import random
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs
from fv3net.diagnostics.prognostic_run.views import movies

GRID = catalog["grid/c48"].read()


def plot_h500(arg, vmin=4800, vmax=6000, vmin_diff=-100, vmax_diff=100):
    ds, filename = arg
    fig, ax = plt.subplots(
        2, 4, figsize=(12, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    real = ds.isel(type=0)
    persistence = ds.isel(type=1)
    gen = ds.isel(type=2)
    c48_real = ds.isel(type=3)
    fv3viz.plot_cube(ds=real, var_name="h500", ax=ax[0, 0], vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("real")
    fv3viz.plot_cube(ds=persistence, var_name="h500", ax=ax[0, 1], vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("persistence")
    fv3viz.plot_cube(ds=gen, var_name="h500", ax=ax[0, 2], vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("gen")
    fv3viz.plot_cube(ds=c48_real, var_name="h500", ax=ax[0, 3], vmin=vmin, vmax=vmax)
    ax[0, 3].set_title("c48_real")
    fv3viz.plot_cube(
        ds=GRID.merge(persistence - real, compat="override"),
        var_name="h500",
        ax=ax[1, 1],
        vmin=vmin_diff,
        vmax=vmax_diff,
    )
    ax[1, 1].set_title("persistence - real")
    fv3viz.plot_cube(
        ds=GRID.merge(gen - real, compat="override"),
        var_name="h500",
        ax=ax[1, 2],
        vmin=vmin_diff,
        vmax=vmax_diff,
    )
    ax[1, 2].set_title("gen - real")
    fv3viz.plot_cube(
        ds=GRID.merge(c48_real - real, compat="override"),
        var_name="h500",
        ax=ax[1, 3],
        vmin=vmin_diff,
        vmax=vmax_diff,
    )
    ax[1, 3].set_title("c48_real - real")
    plt.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def plot_weather(arg):
    ds, filename = arg
    fig, ax = plt.subplots(
        4, 4, figsize=(14, 10), subplot_kw={"projection": ccrs.Robinson()}
    )
    real = ds.isel(type=0)
    persistence = ds.isel(type=1)
    gen = ds.isel(type=2)
    c48_real = ds.isel(type=3)
    vmin = None
    vmax = None
    for i, (varname, vmin, vmax) in enumerate(
        [
            ("h500", 4800, 6000),
            ("PRESsfc", 95e3, 103e3),
            ("TB", 250, 310),
            ("TMP500_300", 210, 270),
        ]
    ):
        fv3viz.plot_cube(
            ds=GRID.merge(real, compat="override"),
            var_name=varname,
            ax=ax[i, 0],
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )
        ax[i, 0].set_title("real")
        fv3viz.plot_cube(
            ds=GRID.merge(persistence, compat="override"),
            var_name=varname,
            ax=ax[i, 1],
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )
        ax[i, 1].set_title("persistence")
        fv3viz.plot_cube(
            ds=GRID.merge(gen, compat="override"),
            var_name=varname,
            ax=ax[i, 2],
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )
        ax[i, 2].set_title("gen")
        fv3viz.plot_cube(
            ds=GRID.merge(c48_real, compat="override"),
            var_name=varname,
            ax=ax[i, 3],
            vmin=vmin,
            vmax=vmax,
        )
        ax[i, 3].set_title("c48_real")

    plt.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    fmr: fv3fit.pytorch.FullModelReplacement = fv3fit.load(
        "model_epoch_0030_loss_0.0909"
    ).to(DEVICE)
    timesteps = 4 * 4
    time_stride = 2
    i_start = 2920
    real_from_disk: xr.Dataset = (
        xr.open_zarr("../cyclegan/c384_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(i_start, i_start + timesteps * time_stride, time_stride))
    ).transpose(..., "time", "tile", "x", "y").load()

    c48_real_from_disk: xr.Dataset = (
        xr.open_zarr("../cyclegan/c48_baseline.zarr")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(0, timesteps * time_stride, time_stride))
    ).transpose(..., "time", "tile", "x", "y").load()
    assert real_from_disk.time[0] == c48_real_from_disk.time[0]
    gen: xr.Dataset
    real: xr.Dataset
    gen, real = fmr.predict(
        real_from_disk.isel(time=range(0, timesteps)), timesteps=timesteps - 1
    )
    gen = gen.isel(window=0)
    real = real.isel(window=0)
    c48_real = c48_real_from_disk.rename({"TMPlowest": "TB"})
    # c48_real = fmr.unpack_tensor(fmr.pack_to_tensor(c48_real_from_disk, timesteps=timesteps - 1)).isel(window=0)
    persistence = xr.concat([real.isel(time=0) for _ in range(timesteps)], dim="time")

    real = real.drop_vars(set(real.data_vars).difference(c48_real.data_vars))
    persistence = persistence.drop_vars(
        set(persistence.data_vars).difference(c48_real.data_vars)
    )
    gen = gen.drop_vars(set(gen.data_vars).difference(c48_real.data_vars))
    c48_real = c48_real.drop_vars(
        set(c48_real.data_vars).difference(real.data_vars)
    ).drop("time")
    ds = xr.concat([real, persistence, gen, c48_real], dim="type").merge(GRID)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    stderr_persistence = (persistence["h500"] - real["h500"]).std(
        dim=["x", "y", "tile"]
    )
    stderr_gen = (gen["h500"] - real["h500"]).std(dim=["x", "y", "tile"])
    stderr_c48 = (c48_real["h500"] - real["h500"]).std(dim=["x", "y", "tile"])
    stderr_persistence.plot(ax=ax, label="persistence")
    stderr_gen.plot(ax=ax, label="generated")
    stderr_c48.plot(ax=ax, label="c48")
    ax.legend(loc="upper left")
    ax.set_ylabel("h500 RMSE")
    plt.tight_layout()
    fig.savefig("h500_weather_stderr.png", dpi=100)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    bias_persistence = (persistence["h500"] - real["h500"]).mean(dim=["x", "y", "tile"])
    bias_gen = (gen["h500"] - real["h500"]).mean(dim=["x", "y", "tile"])
    bias_c48 = (c48_real["h500"] - real["h500"]).mean(dim=["x", "y", "tile"])
    bias_persistence.plot(ax=ax, label="persistence")
    bias_gen.plot(ax=ax, label="generated")
    bias_c48.plot(ax=ax, label="c48")
    ax.legend(loc="upper left")
    ax.set_ylabel("h500 bias")
    plt.tight_layout()
    fig.savefig("h500_weather_bias.png", dpi=100)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.hist(
        real.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="real",
        histtype="step",
        density=True,
    )
    plt.hist(
        gen.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="gen",
        histtype="step",
        density=True,
    )
    plt.hist(
        c48_real.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="c48",
        histtype="step",
        density=True,
    )
    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.xlabel("h500 (Pa)")
    plt.ylabel("probability density")
    plt.tight_layout()
    fig.savefig("h500_histogram_log.png", dpi=100)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.hist(
        real.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="real",
        histtype="step",
        density=True,
    )
    plt.hist(
        gen.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="gen",
        histtype="step",
        density=True,
    )
    plt.hist(
        c48_real.h500.values.flatten(),
        bins=100,
        alpha=0.5,
        label="c48",
        histtype="step",
        density=True,
    )
    plt.legend(loc="upper left")
    plt.xlabel("h500 (Pa)")
    plt.ylabel("probability density")
    plt.tight_layout()
    fig.savefig("h500_histogram.png", dpi=100)

    real_mean = real.mean("time")
    gen_mean = gen.mean("time")
    fig, ax = plt.subplots(
        1, 3, figsize=(12, 5), subplot_kw={"projection": ccrs.Robinson()}
    )
    fv3viz.plot_cube(
        ds=real_mean.merge(GRID), var_name="h500", ax=ax[0], vmin=5000, vmax=5900
    )
    ax[0].set_title("mean(real)")
    fv3viz.plot_cube(
        ds=gen_mean.merge(GRID), var_name="h500", ax=ax[1], vmin=5000, vmax=5900
    )
    ax[1].set_title("mean(gen)")
    fv3viz.plot_cube(
        ds=(gen_mean - real_mean).merge(GRID),
        var_name="h500",
        ax=ax[2],
        vmin=-60,
        vmax=60,
    )
    var = (real_mean - gen_mean).var(dim=["x", "y", "tile"]).h500.values.item()
    mean = (real_mean - gen_mean).mean(dim=["x", "y", "tile"]).h500.values.item()
    ax[2].set_title(
        "mean(gen) - mean(real)\nRMSE: {:.2f}\nbias: {:.2f}".format(var, mean)
    )
    plt.tight_layout()
    fig.savefig("h500_mean.png", dpi=100)

    mse = (real_mean - gen_mean).var()
    var = (real_mean).var()
    print(mse)
    print(var)
    print(1.0 - (mse / var))

    real_std = real.std("time").rename({"h500": "h500_std"})
    gen_std = gen.std("time").rename({"h500": "h500_std"})
    vmin_std = -200
    vmax_std = -vmin_std
    fig, ax = plt.subplots(
        1, 3, figsize=(12, 3), subplot_kw={"projection": ccrs.Robinson()}
    )
    fv3viz.plot_cube(
        ds=real_std.merge(GRID),
        var_name="h500_std",
        ax=ax[0],
        vmin=vmin_std,
        vmax=vmax_std,
    )
    ax[0].set_title("std(real)")
    fv3viz.plot_cube(
        ds=gen_std.merge(GRID),
        var_name="h500_std",
        ax=ax[1],
        vmin=vmin_std,
        vmax=vmax_std,
    )
    ax[1].set_title("std(gen)")
    fv3viz.plot_cube(
        ds=(gen_std - real_std).merge(GRID),
        var_name="h500_std",
        ax=ax[2],
        vmin=vmin_std,
        vmax=vmax_std,
    )
    var = (real_std - gen_std).var(dim=["x", "y", "tile"]).h500_std.values.item()
    mean = (real_std - gen_std).mean(dim=["x", "y", "tile"]).h500_std.values.item()
    ax[2].set_title(
        "std(gen) - std(real)\nRMSE: {:.2f}\nbias: {:.2f}".format(var, mean)
    )
    plt.tight_layout()
    fig.savefig("h500_std.png", dpi=100)

    # fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    # fv3viz.plot_cube(ds=c48_real_std.merge(GRID), var_name="h500", ax=ax[0, 0])
    # ax[0, 0].set_title("c48_real")
    # fv3viz.plot_cube(ds=c384_real_std.merge(GRID), var_name="h500", ax=ax[1, 0])
    # ax[1, 0].set_title("c384_real")
    # fv3viz.plot_cube(ds=c384_gen_std.merge(GRID), var_name="h500", ax=ax[0, 1])
    # ax[0, 1].set_title("c384_gen")
    # fv3viz.plot_cube(ds=c48_gen_std.merge(GRID), var_name="h500", ax=ax[1, 1])
    # ax[1, 1].set_title("c48_gen")
    # plt.tight_layout()

    # fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()})
    # fv3viz.plot_cube(ds=c48_real_std.merge(GRID), var_name="h500", ax=ax[0, 0])
    # ax[0, 0].set_title("c48_real")
    # fv3viz.plot_cube(ds=(c384_real_std - c48_real_std).merge(GRID), var_name="h500", ax=ax[1, 0], vmin=-50, vmax=50)
    # ax[1, 0].set_title("c384_real")
    # fv3viz.plot_cube(ds=(c384_gen_std - c48_real_std).merge(GRID), var_name="h500", ax=ax[0, 1], vmin=-50, vmax=50)
    # ax[0, 1].set_title("c384_gen")
    # fv3viz.plot_cube(ds=(c48_gen_std - c48_real_std).merge(GRID), var_name="h500", ax=ax[1, 1], vmin=-50, vmax=50)
    # ax[1, 1].set_title("c48_gen")
    # plt.tight_layout()
    # fig.savefig("h500_std.png", dpi=100)

    # plt.show()

    spec = movies.MovieSpec(
        name="weather",
        plotting_function=plot_weather,
        required_variables=["h500", "TB", "PRESsfc", "TMP500_300"],
    )
    movies._create_movie(spec, ds, output=".", n_jobs=8)

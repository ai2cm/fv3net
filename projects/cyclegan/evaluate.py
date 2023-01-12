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
from toolz import curry
import numpy as np

GRID = catalog["grid/c48"].read()


class Dataset:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds


class Transformation:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):

        return self.func(*args, **kwargs)

    def compose(self, *args, **kwargs):
        pass


def transformation(func):
    return Transformation(func)


def plot_video():
    pass


def plot_once(ax):
    i_time = random.randint(0, c48_real.time.size - 1)
    fig, ax = plt.subplots(
        2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    fv3viz.plot_cube(ds=c48_real, var_name="h500", ax=ax[0, 0])
    ax[0, 0].set_title("c48_real")
    fv3viz.plot_cube(ds=c384_real, var_name="h500", ax=ax[1, 0])
    ax[1, 0].set_title("c384_real")
    fv3viz.plot_cube(ds=c384_gen, var_name="h500", ax=ax[0, 1])
    ax[0, 1].set_title("c384_gen")
    fv3viz.plot_cube(ds=c48_gen, var_name="h500", ax=ax[1, 1])
    ax[1, 1].set_title("c48_gen")
    plt.tight_layout()


def plot(arg, vmin=4800, vmax=6000):
    ds, filename = arg
    fig, ax = plt.subplots(
        2, 2, figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()}
    )
    c48_real = ds.isel(resolution=0, type=0)
    c384_real = ds.isel(resolution=1, type=0)
    c384_gen = ds.isel(resolution=1, type=1)
    c48_gen = ds.isel(resolution=0, type=1)
    fv3viz.plot_cube(ds=c48_real, var_name="h500", ax=ax[0, 0], vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("c48_real")
    fv3viz.plot_cube(ds=c384_real, var_name="h500", ax=ax[1, 0], vmin=vmin, vmax=vmax)
    ax[1, 0].set_title("c384_real")
    fv3viz.plot_cube(ds=c384_gen, var_name="h500", ax=ax[0, 1], vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("c384_gen")
    fv3viz.plot_cube(ds=c48_gen, var_name="h500", ax=ax[1, 1], vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("c48_gen")
    plt.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def plot_weather(arg, vmin=4800, vmax=6000, vmin_diff=-100, vmax_diff=100):
    ds, filename = arg
    fig, ax = plt.subplots(
        2, 3, figsize=(12, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    c48_real = ds.isel(resolution=0, type=0)
    c384_real = ds.isel(resolution=1, type=0)
    c384_gen = ds.isel(resolution=1, type=1)
    fv3viz.plot_cube(ds=c384_real, var_name="h500", ax=ax[0, 0], vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("c384_real")
    fv3viz.plot_cube(ds=c48_real, var_name="h500", ax=ax[0, 1], vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("c48_real")
    fv3viz.plot_cube(ds=c384_gen, var_name="h500", ax=ax[0, 2], vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("c384_gen")
    fv3viz.plot_cube(
        ds=GRID.merge(c48_real - c384_real, compat="override"),
        var_name="h500",
        ax=ax[1, 1],
        vmin=vmin_diff,
        vmax=vmax_diff,
    )
    ax[1, 1].set_title("c48_real - c384_real")
    fv3viz.plot_cube(
        ds=GRID.merge(c384_gen - c384_real, compat="override"),
        var_name="h500",
        ax=ax[1, 2],
        vmin=vmin_diff,
        vmax=vmax_diff,
    )
    ax[1, 2].set_title("c384_gen - c384_real")
    plt.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        "gs://vcm-ml-experiments/cyclegan/2023-01-11/cyclegan_c48_to_c384-trial-0/"
    ).to(DEVICE)
    c384_real: xr.Dataset = (
        xr.open_zarr("./fine-0K.zarr/")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(2920, None))
        .load()
    )
    c48_real: xr.Dataset = (
        xr.open_zarr("./coarse-0K.zarr/")
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .isel(time=slice(11688, None))
        .load()
    )

    c384_gen: xr.Dataset = cyclegan.predict(c48_real)
    c48_gen: xr.Dataset = cyclegan.predict(c384_real, reverse=True)
    if "PRATEsfc_log" in c384_gen:
        c384_gen["PRATEsfc"] = np.exp(c384_gen["PRATEsfc_log"]) - 0.00003
        c48_gen["PRATEsfc"] = np.exp(c48_gen["PRATEsfc_log"]) - 0.00003
        c384_real["PRATEsfc"] = np.exp(c384_real["PRATEsfc_log"]) - 0.00003
        c48_real["PRATEsfc"] = np.exp(c48_real["PRATEsfc_log"]) - 0.00003

    # ds = xr.concat(
    #     [
    #         xr.concat([c48_real.drop("time"), c384_real.drop("time")], dim="resolution"),
    #         xr.concat([c48_gen, c384_gen], dim="resolution")
    #     ], dim="type"
    # ).merge(GRID)

    # spec = movies.MovieSpec(name="h500", plotting_function=plot, required_variables=["h500"])
    # movies._create_movie(spec, ds.isel(time=range(0, 4*30)), output=".", n_jobs=8)

    # spec = movies.MovieSpec(
    #     name="h500_weather",
    #     plotting_function=plot_weather,
    #     required_variables=["h500"]
    # )
    # movies._create_movie(spec, ds.isel(time=range(0, 4*7)), output=".", n_jobs=8)

    # fig, ax = plt.subplots(
    #     1, 1, figsize=(5, 3)
    # )
    # stderr_baseline = (
    #     c48_real["h500"].isel(time=range(0, 4*7)) - c384_real["h500"].isel(time=range(0, 4*7))
    # ).std(dim=["x", "y", "tile"])
    # stderr_gen = (
    #     c384_gen["h500"].isel(time=range(0, 4*7)) - c384_real["h500"].isel(time=range(0, 4*7))
    # ).std(dim=["x", "y", "tile"])
    # stderr_baseline.plot(ax=ax, label="baseline")
    # stderr_gen.plot(ax=ax, label="generated")
    # ax.legend(loc="upper left")
    # ax.set_ylabel("h500 standard error vs c384_real")
    # plt.tight_layout()
    # fig.savefig("h500_weather_stderr.png", dpi=100)

    # fig, ax = plt.subplots(
    #     1, 1, figsize=(5, 3)
    # )
    # bias_baseline = (
    #     c48_real["h500"].isel(time=range(0, 4*7)) - c384_real["h500"].isel(time=range(0, 4*7))
    # ).mean(dim=["x", "y", "tile"])
    # bias_gen = (
    #     c384_gen["h500"].isel(time=range(0, 4*7)) - c384_real["h500"].isel(time=range(0, 4*7))
    # ).mean(dim=["x", "y", "tile"])
    # bias_baseline.plot(ax=ax, label="baseline")
    # bias_gen.plot(ax=ax, label="generated")
    # ax.legend(loc="upper left")
    # ax.set_ylabel("h500 bias vs c384_real")
    # plt.tight_layout()
    # fig.savefig("h500_weather_bias.png", dpi=100)

    def plot_hist(varname, units, log: bool):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        plt.hist(
            c48_real[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label="c48_real",
            histtype="step",
            density=True,
        )
        plt.hist(
            c384_real[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label="c384_real",
            histtype="step",
            density=True,
        )
        plt.hist(
            c384_gen[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label=f"c384_gen",
            histtype="step",
            density=True,
        )
        if log:
            plt.yscale("log")
            log_suffix = "_log"
        else:
            log_suffix = ""
        # plt.hist(c48_gen[varname].values.flatten(), bins=100, alpha=0.5, label="c48_gen")
        plt.legend(loc="upper left")
        plt.xlabel(f"{varname} ({units})")
        plt.ylabel("probability density")
        plt.tight_layout()
        fig.savefig(f"{varname}_histogram{log_suffix}.png", dpi=100)

    plot_hist("h500", "m", log=False)
    plot_hist("h500", "m", log=True)
    plot_hist("PRATEsfc", "kg/m^2/s", log=False)
    plot_hist("PRATEsfc", "kg/m^2/s", log=True)

    c48_real_mean = c48_real.mean("time")
    c48_gen_mean = c48_gen.mean("time")
    c384_real_mean = c384_real.mean("time")
    c384_gen_mean = c384_gen.mean("time")

    def plot_mean(varname: str, vmin: float, vmax: float):
        fig, ax = plt.subplots(
            2, 2, figsize=(10, 6), subplot_kw={"projection": ccrs.Robinson()}
        )
        fv3viz.plot_cube(
            ds=GRID.merge(c48_real_mean, compat="override"),
            var_name=varname,
            ax=ax[0, 0],
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, 0].set_title("c48_real")
        fv3viz.plot_cube(
            ds=GRID.merge(c384_real_mean, compat="override"),
            var_name=varname,
            ax=ax[1, 0],
            vmin=vmin,
            vmax=vmax,
        )
        ax[1, 0].set_title("c384_real")
        fv3viz.plot_cube(
            ds=GRID.merge(c384_gen_mean, compat="override"),
            var_name=varname,
            ax=ax[0, 1],
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, 1].set_title("c384_gen")
        fv3viz.plot_cube(
            ds=GRID.merge(c48_gen_mean, compat="override"),
            var_name=varname,
            ax=ax[1, 1],
            vmin=vmin,
            vmax=vmax,
        )
        ax[1, 1].set_title("c48_gen")
        plt.tight_layout()
        fig.savefig(f"{varname}_mean.png", dpi=100)

    plot_mean("h500", vmin=5000, vmax=5900)

    def plot_bias(varname, vmin_abs, vmax_abs, vmin_rel, vmax_rel):
        fig, ax = plt.subplots(
            1, 3, figsize=(12, 3), subplot_kw={"projection": ccrs.Robinson()}
        )
        fv3viz.plot_cube(
            ds=GRID.merge(c384_real_mean, compat="override"),
            var_name=varname,
            ax=ax[0],
            vmin=vmin_abs,
            vmax=vmax_abs,
        )
        ax[0].set_title("c384_real")
        fv3viz.plot_cube(
            ds=GRID.merge(c48_real_mean - c384_real_mean, compat="override"),
            var_name=varname,
            ax=ax[1],
            vmin=vmin_rel,
            vmax=vmax_rel,
        )
        var = (
            (c48_real_mean - c384_real_mean)
            .var(dim=["x", "y", "tile"])[varname]
            .values.item()
        )
        mean = (
            (c48_real_mean - c384_real_mean)
            .mean(dim=["x", "y", "tile"])[varname]
            .values.item()
        )
        ax[1].set_title(
            "c48_real - c384_real\nvar: {:.2f}\nmean: {:.2f}".format(var, mean)
        )
        fv3viz.plot_cube(
            ds=GRID.merge(c384_gen_mean - c384_real_mean, compat="override"),
            var_name=varname,
            ax=ax[2],
            vmin=vmin_rel,
            vmax=vmax_rel,
        )
        var = (
            (c384_gen_mean - c384_real_mean)
            .var(dim=["x", "y", "tile"])[varname]
            .values.item()
        )
        mean = (
            (c384_gen_mean - c384_real_mean)
            .mean(dim=["x", "y", "tile"])[varname]
            .values.item()
        )
        ax[2].set_title(
            "c384_gen - c384_real\nvar = {:.2f}\nmean = {:.2f}".format(var, mean)
        )
        plt.tight_layout()
        fig.savefig(f"{varname}_mean_diff.png", dpi=100)

    plot_bias("h500", vmin_abs=4800, vmax_abs=4800, vmin_rel=-100, vmax_rel=100)

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

    c48_real_std = c48_real.std("time")
    c48_gen_std = c48_gen.std("time")
    c384_real_std = c384_real.std("time")
    c384_gen_std = c384_gen.std("time")

    def plot_std(varname, vmin: float, vmax: float):
        fig, ax = plt.subplots(
            1, 3, figsize=(12, 3), subplot_kw={"projection": ccrs.Robinson()}
        )
        c48_real_renamed = c48_real_std.rename({varname: f"{varname}_std"})
        c48_gen_renamed = c48_gen_std.rename({varname: f"{varname}_std"})
        c384_real_renamed = c384_real_std.rename({varname: f"{varname}_std"})
        c384_gen_renamed = c384_gen_std.rename({varname: f"{varname}_std"})
        fv3viz.plot_cube(
            ds=GRID.merge(c48_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[0],
        )
        ax[0].set_title("c48_real")
        fv3viz.plot_cube(
            ds=GRID.merge(c48_gen_renamed - c48_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[1],
            vmin=vmin,
            vmax=vmax,
        )
        var = (
            (c48_gen_renamed - c48_real_renamed)
            .var(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        mean = (
            (c48_gen_renamed - c48_real_renamed)
            .mean(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        ax[1].set_title(
            "c48_gen - c48_real\nvar: {:.2f}\nmean: {:.2f}".format(var, mean)
        )
        fv3viz.plot_cube(
            ds=GRID.merge(c384_gen_renamed - c384_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[2],
            vmin=vmin,
            vmax=vmax,
        )
        var = (
            (c384_gen_renamed - c384_real_renamed)
            .var(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        mean = (
            (c384_gen_renamed - c384_real_renamed)
            .mean(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        ax[2].set_title(
            "c384_gen - c384_real\nvar = {:.2f}\nmean = {:.2f}".format(var, mean)
        )
        plt.tight_layout()
        fig.savefig(f"{varname}_std_diff.png", dpi=100)
        fv3viz.plot_cube(
            ds=GRID.merge(c384_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[0],
        )
        ax[0].set_title("c384_real")
        fv3viz.plot_cube(
            ds=GRID.merge(c48_real_renamed - c384_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[1],
            vmin=vmin,
            vmax=vmax,
        )
        var = (
            (c48_real_renamed - c384_real_renamed)
            .var(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        mean = (
            (c48_real_renamed - c384_real_renamed)
            .mean(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        ax[1].set_title(
            "c48_real - c384_real\nvar = {:.2f}\nmean = {:.2f}".format(var, mean)
        )
        fv3viz.plot_cube(
            ds=GRID.merge(c384_gen_renamed - c384_real_renamed, compat="override"),
            var_name=f"{varname}_std",
            ax=ax[2],
            vmin=vmin,
            vmax=vmax,
        )
        var = (
            (c384_gen_renamed - c384_real_renamed)
            .var(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        mean = (
            (c384_gen_renamed - c384_real_renamed)
            .mean(dim=["x", "y", "tile"])[f"{varname}_std"]
            .values.item()
        )
        ax[2].set_title(
            "c384_gen - c384_real\nvar = {:.2f}\nmean = {:.2f}".format(var, mean)
        )
        plt.tight_layout()
        fig.savefig(f"{varname}_std_diff.png", dpi=100)

    plot_std("h500", vmin=-60, vmax=60)

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

    plt.show()

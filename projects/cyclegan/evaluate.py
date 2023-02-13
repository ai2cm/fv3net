# flake8: noqa

import random
from typing import List
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
import sklearn.preprocessing

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


def cross_entropy(x1, x2):
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=1000, subsample=100000
    )
    transformer.fit(x1.reshape(-1, 1))
    quantiles = transformer.quantiles_.copy()[:, 0]
    quantiles[0] = min(x1.min(), x2.min())
    quantiles[-1] = max(x1.max(), x2.max())
    x1_hist = np.histogram(x1, bins=quantiles)[0]
    x2_hist = np.histogram(x2, bins=quantiles)[0]
    x1_hist = x1_hist / x1_hist.sum()
    x2_hist = x2_hist / x2_hist.sum()
    x1_hist[x1_hist == 0] = 1e-20
    x2_hist[x2_hist == 0] = 1e-20
    bin_scales = (quantiles[1:] - quantiles[:-1]) / (quantiles[-1] - quantiles[0])
    return -np.sum(x1_hist * np.log(x2_hist) * bin_scales)


def evaluate(
    cyclegan: fv3fit.pytorch.CycleGAN,
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    expected_bias_c384: xr.Dataset,
):
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

    def plot_hist_all():
        varnames = cyclegan.state_variables
        fig, ax = plt.subplots(len(varnames), 2, figsize=(10, 1 + 2.5 * len(varnames)))
        if len(varnames) == 1:
            ax = ax[None, :]
        for i, varname in enumerate(varnames):
            plot_hist(varname, ax=ax[i, 0], log=False)
            plot_hist(varname, ax=ax[i, 1], log=True)
        plt.tight_layout()
        fig.savefig(f"histogram.png", dpi=100)

    def plot_hist(varname, log: bool, units=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        else:
            fig = None
        ax.hist(
            c48_real[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label="c48_real",
            histtype="step",
            density=True,
        )
        ax.hist(
            c384_real[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label="c384_real",
            histtype="step",
            density=True,
        )
        ax.hist(
            c384_gen[varname].values.flatten(),
            bins=100,
            alpha=0.5,
            label=f"c384_gen",
            histtype="step",
            density=True,
        )
        gen_cross_entropy = cross_entropy(
            c384_real[varname].values.flatten(), c384_gen[varname].values.flatten()
        )
        c48_cross_entropy = cross_entropy(
            c384_real[varname].values.flatten(), c48_real[varname].values.flatten()
        )

        if log:
            ax.set_yscale("log")
            log_suffix = "_log"
        else:
            log_suffix = ""
        # plt.hist(c48_gen[varname].values.flatten(), bins=100, alpha=0.5, label="c48_gen")
        ax.legend(loc="upper left")
        if units is not None:
            ax.set_xlabel(f"{varname} ({units})")
        else:
            ax.set_xlabel(varname)
        ax.set_ylabel("probability density")
        ax.set_title(
            f"{varname}\nH(p, q): {gen_cross_entropy:.2e} (gen), {c48_cross_entropy:.2e} (c48)"
        )
        if fig is not None:
            plt.tight_layout()
            fig.savefig(f"{varname}_histogram{log_suffix}.png", dpi=100)

    plot_hist_all()
    # plot_hist("h500", units="m", log=False)
    # plot_hist("h500", units="m", log=True)
    # plot_hist("PRATEsfc", units="kg/m^2/s", log=False)
    # plot_hist("PRATEsfc", units="kg/m^2/s", log=True)

    c48_real_mean = c48_real.mean("time")
    c48_gen_mean = c48_gen.mean("time")
    c384_real_mean = c384_real.mean("time")
    c384_gen_mean = c384_gen.mean("time")

    def plot_mean_all():
        varnames = cyclegan.state_variables
        fig, ax = plt.subplots(
            len(varnames),
            4,
            figsize=(18, 3.5 * len(varnames)),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        if len(varnames) == 1:
            ax = ax[None, :]
        real = GRID.merge(c384_real_mean, compat="override")
        gen = GRID.merge(c384_gen_mean, compat="override")
        for i, varname in enumerate(varnames):
            gen[f"{varname}_bias"] = (
                gen[varname] - real[varname] - expected_bias_c384[varname]
            )
            gen[f"{varname}_c48_bias"] = (
                c48_real_mean[varname] - real[varname] - expected_bias_c384[varname]
            )
            vmin = min(
                c384_real_mean[varname].min().values,
                c384_gen_mean[varname].min().values,
            )
            vmax = max(
                c384_real_mean[varname].max().values,
                c384_gen_mean[varname].max().values,
            )
            ax[i, 0].set_title("c384_real")
            fv3viz.plot_cube(
                ds=GRID.merge(c384_real_mean, compat="override"),
                var_name=varname,
                ax=ax[i, 0],
                vmin=vmin,
                vmax=vmax,
            )
            ax[i, 1].set_title("c384_gen")
            fv3viz.plot_cube(
                ds=GRID.merge(c384_gen_mean, compat="override"),
                var_name=varname,
                ax=ax[i, 1],
                vmin=vmin,
                vmax=vmax,
            )
            bias_max = max(
                gen[f"{varname}_bias"].max().values,
                gen[f"{varname}_c48_bias"].max().values,
                -gen[f"{varname}_bias"].min().values,
                -gen[f"{varname}_c48_bias"].min().values,
            )
            gen_bias_mean = gen[f"{varname}_bias"].values.mean()
            gen_bias_std = gen[f"{varname}_bias"].values.std()
            c48_bias_mean = gen[f"{varname}_c48_bias"].values.mean()
            c48_bias_std = gen[f"{varname}_c48_bias"].values.std()
            fv3viz.plot_cube(
                ds=gen,
                var_name=f"{varname}_bias",
                ax=ax[i, 2],
                vmin=-bias_max,
                vmax=bias_max,
            )
            ax[i, 2].set_title(
                "gen_bias\nmean: {:.2e}\nstd: {:.2e}".format(
                    gen_bias_mean, gen_bias_std
                )
            )
            fv3viz.plot_cube(
                ds=gen,
                var_name=f"{varname}_c48_bias",
                ax=ax[i, 3],
                vmin=-bias_max,
                vmax=bias_max,
            )
            ax[i, 3].set_title(
                "c48_bias\nmean: {:.2e}\nstd: {:.2e}".format(
                    c48_bias_mean, c48_bias_std
                )
            )

        plt.tight_layout()
        fig.savefig(f"mean.png", dpi=100)

    plot_mean_all()
    # plot_mean("h500", vmin=5000, vmax=5900)
    # plot_mean("PRATEsfc", vmin=None, vmax=None)

    def plot_mean_all_reverse():
        varnames = cyclegan.state_variables
        fig, ax = plt.subplots(
            len(varnames),
            4,
            figsize=(14, 2 * len(varnames)),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        if len(varnames) == 1:
            ax = ax[None, :]
        real = GRID.merge(c48_real_mean, compat="override")
        gen = GRID.merge(c48_gen_mean, compat="override")
        for i, varname in enumerate(varnames):
            gen[f"{varname}_bias"] = gen[varname] - real[varname]
            gen[f"{varname}_c384_bias"] = gen[varname] - c384_real_mean[varname]
            vmin = min(
                c48_real_mean[varname].min().values, c48_gen_mean[varname].min().values,
            )
            vmax = max(
                c48_real_mean[varname].max().values, c48_gen_mean[varname].max().values,
            )
            fv3viz.plot_cube(
                ds=GRID.merge(c48_gen_mean, compat="override"),
                var_name=varname,
                ax=ax[i, 0],
                vmin=vmin,
                vmax=vmax,
            )
            fv3viz.plot_cube(
                ds=GRID.merge(c384_gen_mean, compat="override"),
                var_name=varname,
                ax=ax[i, 1],
                vmin=vmin,
                vmax=vmax,
            )
            bias_max = max(
                gen[f"{varname}_bias"].max().values,
                gen[f"{varname}_c384_bias"].max().values,
                -gen[f"{varname}_bias"].min().values,
                -gen[f"{varname}_c384_bias"].min().values,
            )
            fv3viz.plot_cube(
                ds=gen,
                var_name=f"{varname}_bias",
                ax=ax[i, 2],
                vmin=-bias_max,
                vmax=bias_max,
            )
            fv3viz.plot_cube(
                ds=gen,
                var_name=f"{varname}_c384_bias",
                ax=ax[i, 3],
                vmin=-bias_max,
                vmax=bias_max,
            )
        ax[0, 0].set_title("c48_real")
        ax[0, 1].set_title("c48_gen")
        ax[0, 2].set_title("gen_bias")
        ax[0, 3].set_title("c384_bias")

        plt.tight_layout()
        fig.savefig(f"mean_reverse.png", dpi=100)

    # plot_mean_all_reverse()

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

    # plot_bias("h500", vmin_abs=4800, vmax_abs=4800, vmin_rel=-100, vmax_rel=100)
    # plot_bias("PRATEsfc", None, None, None, None)

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

    # plot_std("h500", vmin=-60, vmax=60)

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


def plot_annual_means(ds: xr.Dataset, varnames: List[str]):
    # we will take the mean of the dataset for each year and plot each of those
    # in a different panel with matplotlib
    # dataset is 3-hourly so there are 365 * 8 = 2920 timesteps per year

    n_years = int(len(ds.time) / 2920)
    assert n_years > 0
    fig, ax = plt.subplots(
        n_years,
        len(varnames),
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    for i in range(n_years):
        ds_year = ds.isel(time=slice(i * 2920, (i + 1) * 2920)).mean(dim="time")
        for j, varname in enumerate(varnames):
            fv3viz.plot_cube(
                ds=GRID.merge(ds_year, compat="override"), var_name=varname, ax=ax[i, j]
            )
            ax[i, j].set_title(f"{varname} year {i + 1}")
    plt.tight_layout()
    fig.savefig(f"annual_means.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    random.seed(0)
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        # "gs://vcm-ml-experiments/cyclegan/2023-01-20/cyclegan_c48_to_c384-prec-h500-w512-2e-4"  # prec/h500, 512-width, epoch 50
        # "gs://vcm-ml-experiments/cyclegan/2023-01-22/cyclegan_c48_to_c384-prec-h500-w512-2e-4-epoch50" # prec/h500, 512-width, epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-211721-5ddf5ef3-epoch_040/"  # reduced-vars, 512-width
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-174516-bfffae02-epoch_025/"  # h500-only
        # "gs://vcm-ml-experiments/cyclegan/2023-01-19/cyclegan_c48_to_c384-h500only-kernel4-2e-4"  # h500-only, epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-171215-64882f40-epoch_060/"  # precip-only
        # "gs://vcm-ml-experiments/cyclegan/2023-01-19/cyclegan_c48_to_c384-preconly-kernel4-2e-4"  # precip-only epoch 100
        "gs://vcm-ml-experiments/cyclegan/2023-01-22/cyclegan_c48_to_c384-preconly-kernel4-2e-5-epoch50"  # precip-only, 2e-5, epoch 200
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230122-163054-99bf1aed-epoch_066/"  # precip-only, 2e-5 +100 epochs
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-173808-a5fd4151-epoch_051/"  # 2e-4, kernel4
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-183240-b191e306-epoch_048/"  # 2e-4, kernel3
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-003753-f8afe719-epoch_085/"  # log, 2e-5
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-003758-1324b685-epoch_035/"  # log, 2e-4
        # "gs://vcm-ml-experiments/cyclegan/2023-01-11/cyclegan_c48_to_c384-trial-0/"
        # "gs://vcm-ml-experiments/cyclegan/2023-02-02/cyclegan_c48_to_c384-precip-h500-2e-5"  # precip/h500, 2e-5, epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230203-191435-7411d5db-epoch_137/"  # 2-precip 2e-5 (3x3 gen kernel) train-cyclegan-8170caad17cf
    ).to(DEVICE)

    subselect_ratio = 1.0 / 4
    c384_real_all: xr.Dataset = (
        xr.open_zarr("./fine-0K.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c48_real_all: xr.Dataset = (
        xr.open_zarr("./coarse-0K.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c384_real_all["PRATEsfc_2"] = c384_real_all["PRATEsfc"]
    c48_real_all["PRATEsfc_2"] = c48_real_all["PRATEsfc"]

    c384_real: xr.Dataset = c384_real_all.isel(time=slice(2920, None))
    c384_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c384_real.time)),
            size=int(len(c384_real.time) * subselect_ratio),
            replace=False,
        )
    )
    c384_real = c384_real.isel(time=c384_steps).load()
    c48_real: xr.Dataset = c48_real_all.isel(time=slice(11688, None))
    c48_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c48_real.time)),
            size=int(len(c48_real.time) * subselect_ratio),
            replace=False,
        )
    )
    c48_real = c48_real.isel(time=c48_steps).load()

    train_c48_mean = c48_real_all.isel(time=slice(0, 11688)).mean(dim="time")
    val_c48_mean = c48_real_all.isel(time=slice(11688, None)).mean(dim="time")
    train_c384_mean = c384_real_all.isel(time=slice(0, 2920)).mean(dim="time")
    val_c384_mean = c384_real_all.isel(time=slice(2920, None)).mean(dim="time")
    expected_bias_c384 = train_c384_mean - val_c384_mean
    expected_bias_c48 = train_c48_mean - val_c48_mean
    expected_bias = expected_bias_c48

    # plot the expected bias for each state variable of the cyclegan

    fig, ax = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.Robinson()}
    )
    if "h500" in expected_bias.data_vars:
        fv3viz.plot_cube(
            ds=GRID.merge(expected_bias, compat="override"),
            var_name="h500",
            ax=ax[0],
            vmin=-150,
            vmax=150,
        )
    if "PRATEsfc" in expected_bias.data_vars:
        fv3viz.plot_cube(
            ds=GRID.merge(expected_bias, compat="override"),
            var_name="PRATEsfc",
            ax=ax[1],
            vmin=-1e-4,
            vmax=1e-4,
        )
    plt.tight_layout()
    fig.savefig(f"expected_bias.png", dpi=100)

    # plot_annual_means(c48_real, cyclegan.state_variables)
    evaluate(cyclegan, c48_real, c384_real, expected_bias_c384)

# flake8: noqa

import random
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs
import numpy as np

GRID = catalog["grid/c48"].read()


def evaluate(
    cyclegan: fv3fit.pytorch.CycleGAN,
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    varname: str,
):
    c384_list = []
    c48_list = []
    for i in range(len(c384_real.perturbation)):
        c384_list.append(cyclegan.predict(c48_real.isel(perturbation=i)))
        c48_list.append(cyclegan.predict(c384_real.isel(perturbation=i), reverse=True))
    c384_gen = xr.concat(c384_list, dim="perturbation")
    c48_gen = xr.concat(c48_list, dim="perturbation")
    # c384_gen: xr.Dataset = cyclegan.predict(c48_real)
    # c48_gen: xr.Dataset = cyclegan.predict(c384_real, reverse=True)
    # if "PRATEsfc_log" in c384_gen:
    #     c384_gen["PRATEsfc"] = np.exp(c384_gen["PRATEsfc_log"]) - 0.00003
    #     c48_gen["PRATEsfc"] = np.exp(c48_gen["PRATEsfc_log"]) - 0.00003
    #     c384_real["PRATEsfc"] = np.exp(c384_real["PRATEsfc_log"]) - 0.00003
    #     c48_real["PRATEsfc"] = np.exp(c48_real["PRATEsfc_log"]) - 0.00003

    def plot_hist_all(varname):
        fig, ax = plt.subplots(
            len(c384_real.perturbation),
            2,
            figsize=(10, 1 + 2.5 * len(c384_real.perturbation)),
        )
        for i, climate in enumerate(c384_real.perturbation.values):
            plot_hist(i, varname, ax=ax[i, 0], log=False)
            plot_hist(i, varname, ax=ax[i, 1], log=True)
            ax[i, 0].set_title(f"{climate} (linear)")
            ax[i, 1].set_title(f"{climate} (log)")
        plt.tight_layout()
        fig.savefig(f"histogram.png", dpi=100)

    def plot_hist(i_perturbation, varname, log: bool, units=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        else:
            fig = None
        ax.hist(
            c48_real[varname].isel(perturbation=i_perturbation).values.flatten(),
            bins=100,
            alpha=0.5,
            label="c48_real",
            histtype="step",
            density=True,
        )
        ax.hist(
            c384_real[varname].isel(perturbation=i_perturbation).values.flatten(),
            bins=100,
            alpha=0.5,
            label="c384_real",
            histtype="step",
            density=True,
        )
        ax.hist(
            c384_gen[varname].isel(perturbation=i_perturbation).values.flatten(),
            bins=100,
            alpha=0.5,
            label=f"c384_gen",
            histtype="step",
            density=True,
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
        if fig is not None:
            plt.tight_layout()
            fig.savefig(f"{varname}_histogram{log_suffix}.png", dpi=100)

    plot_hist_all(varname)

    c48_real_mean = c48_real.mean("time")
    c48_gen_mean = c48_gen.mean("time")
    c384_real_mean = c384_real.mean("time")
    c384_gen_mean = c384_gen.mean("time")

    def plot_mean_all(varname):
        fig, ax = plt.subplots(
            len(c384_real.perturbation),
            4,
            figsize=(18, 3.5 * len(c384_real.perturbation)),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        real = GRID.merge(c384_real_mean, compat="override")
        gen = GRID.merge(c384_gen_mean, compat="override")
        gen[f"{varname}_bias"] = gen[varname] - real[varname]
        gen[f"{varname}_c48_bias"] = c48_real_mean[varname] - real[varname]
        # vmin = min(
        #     c384_real_mean[varname].min().values,
        #     c384_gen_mean[varname].min().values,
        # )
        # vmax = max(
        #     c384_real_mean[varname].max().values,
        #     c384_gen_mean[varname].max().values,
        # )
        vmin, vmax = None, None
        bias_max = max(
            gen[f"{varname}_bias"].max().values,
            gen[f"{varname}_c48_bias"].max().values,
            -gen[f"{varname}_bias"].min().values,
            -gen[f"{varname}_c48_bias"].min().values,
        )
        for i, climate in enumerate(c384_real.perturbation.values):
            ax[i, 0].set_title(f"{climate} c384_real")
            fv3viz.plot_cube(
                ds=GRID.merge(c384_real_mean.isel(perturbation=i), compat="override"),
                var_name=varname,
                ax=ax[i, 0],
                vmin=vmin,
                vmax=vmax,
            )
            ax[i, 1].set_title(f"{climate} c384_gen")
            fv3viz.plot_cube(
                ds=GRID.merge(c384_gen_mean.isel(perturbation=i), compat="override"),
                var_name=varname,
                ax=ax[i, 1],
                vmin=vmin,
                vmax=vmax,
            )
            gen_bias_mean = gen[f"{varname}_bias"].isel(perturbation=i).values.mean()
            gen_bias_std = gen[f"{varname}_bias"].isel(perturbation=i).values.std()
            c48_bias_mean = (
                gen[f"{varname}_c48_bias"].isel(perturbation=i).values.mean()
            )
            c48_bias_std = gen[f"{varname}_c48_bias"].isel(perturbation=i).values.std()
            fv3viz.plot_cube(
                ds=gen.isel(perturbation=i),
                var_name=f"{varname}_bias",
                ax=ax[i, 2],
                # vmin=-bias_max,
                # vmax=bias_max,
            )
            ax[i, 2].set_title(
                "{} gen_bias\nmean: {:.2e}\nstd: {:.2e}".format(
                    climate, gen_bias_mean, gen_bias_std
                )
            )
            fv3viz.plot_cube(
                ds=gen.isel(perturbation=i),
                var_name=f"{varname}_c48_bias",
                ax=ax[i, 3],
                # vmin=-bias_max,
                # vmax=bias_max,
            )
            ax[i, 3].set_title(
                "{} c48_bias\nmean: {:.2e}\nstd: {:.2e}".format(
                    climate, c48_bias_mean, c48_bias_std
                )
            )

        plt.tight_layout()
        fig.savefig(f"mean.png", dpi=100)

    plot_mean_all(varname)

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

    plt.show()


if __name__ == "__main__":
    random.seed(0)
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230130-231729-82b939d9-epoch_033/"  # precip-only
    ).to(DEVICE)

    subselect_ratio = 1.0 / 256
    c384_real_all: xr.Dataset = (
        xr.open_zarr(
            "gs://vcm-ml-experiments/mcgibbon/2023-01-27/fine-combined.zarr/"
        ).rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c48_real_all: xr.Dataset = (
        xr.open_zarr(
            "gs://vcm-ml-experiments/mcgibbon/2023-01-27/coarse-combined.zarr/"
        ).rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c384_real: xr.Dataset = c384_real_all.isel(time=slice(2920, None))
    c384_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c384_real.time)),
            size=int(len(c384_real.time) * subselect_ratio),
            replace=False,
        )
    )
    # c384_steps=[0]
    c384_real = c384_real.isel(time=c384_steps).load()
    c48_real: xr.Dataset = c48_real_all.isel(time=slice(11688, None))
    c48_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c48_real.time)),
            size=int(len(c48_real.time) * subselect_ratio),
            replace=False,
        )
    )
    # c48_steps=[0]
    c48_real = c48_real.isel(time=c48_steps).load()

    # train_c48_mean = c48_real_all.isel(time=slice(0, 11688)).mean(dim="time")
    # val_c48_mean = c48_real_all.isel(time=slice(11688, None)).mean(dim="time")
    # train_c384_mean = c384_real_all.isel(time=slice(0, 2920)).mean(dim="time")
    # val_c384_mean = c384_real_all.isel(time=slice(2920, None)).mean(dim="time")
    # expected_bias_c384 = train_c384_mean - val_c384_mean
    # expected_bias_c48 = train_c48_mean - val_c48_mean
    # expected_bias = expected_bias_c48

    # # plot the expected bias for each state variable of the cyclegan

    # fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.Robinson()})
    # if "h500" in expected_bias.data_vars:
    #     fv3viz.plot_cube(
    #         ds=GRID.merge(expected_bias, compat="override"),
    #         var_name="h500",
    #         ax=ax[0],
    #         vmin=-150,
    #         vmax=150,
    #     )
    # if "PRATEsfc" in expected_bias.data_vars:
    #     fv3viz.plot_cube(
    #         ds=GRID.merge(expected_bias, compat="override"),
    #         var_name="PRATEsfc",
    #         ax=ax[1],
    #         vmin=-1e-4,
    #         vmax=1e-4,
    #     )
    # plt.tight_layout()
    # fig.savefig(f"expected_bias.png", dpi=100)

    # plot_annual_means(c48_real, cyclegan.state_variables)
    evaluate(cyclegan, c48_real, c384_real, varname="PRATEsfc")

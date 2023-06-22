# flake8: noqa

import functools
import os
import random
from typing import Optional, Tuple
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs
import numpy as np
import sklearn.preprocessing
import pickle

GRID = catalog["grid/c48"].read()


def cross_entropy(x1, x2):
    vmin = min(x1.min(), x2.min())
    vmax = max(x1.max(), x2.max())
    bin_edges = np.linspace(vmin, vmax, min(x1.shape[0] / 1000, 1000))
    x1_hist = np.histogram(x1, bins=bin_edges)[0]
    x2_hist = np.histogram(x2, bins=bin_edges)[0]
    x1_hist = x1_hist / x1_hist.sum()
    x2_hist = x2_hist / x2_hist.sum()
    x1_hist[x1_hist == 0] = 1e-20
    x2_hist[x2_hist == 0] = 1e-20
    bin_scales = (bin_edges[1:] - bin_edges[:-1]) / (bin_edges[-1] - bin_edges[0])
    return -np.sum(x1_hist * np.log(x2_hist) * bin_scales)


class QuantileMapping:
    def __init__(self, transform_in, transform_out):
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __call__(self, x):
        return self.transform_out.inverse_transform(self.transform_in.transform(x))

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class Data:
    def __init__(
        self,
        c48: xr.Dataset,
        c384: xr.Dataset,
        varname,
        transform_path: str,
        cyclegan: fv3fit.pytorch.CycleGAN,
        subselect_ratio: float = 1.0 / 4,
    ):
        c48_all = c48.load()
        c384_all = c384.load()
        self._c48_train = c48_all.isel(time=slice(0, 11688))
        self._c384_train = c384_all.isel(time=slice(0, 2920))
        self._c48_test = c48_all.isel(time=slice(11688, None))
        self._c384_test = c384_all.isel(time=slice(2920, None))
        self.varname = varname
        self.cyclegan = cyclegan

        if os.path.exists(transform_path):
            with open(transform_path, "rb") as f:
                self.transform: QuantileMapping = pickle.load(f)
        else:
            self.transform = train_quantile_mapping(
                self._c48_train, self._c384_train, self.varname,
            )
            self.transform.dump(transform_path)

        self._c48_steps = np.sort(
            np.random.choice(
                np.arange(0, len(self._c48_test.time)),
                size=int(len(self._c48_test.time) * subselect_ratio),
                replace=False,
            )
        )
        self._c384_steps = np.sort(
            np.random.choice(
                np.arange(0, len(self._c384_test.time)),
                size=int(len(self._c384_test.time) * subselect_ratio),
                replace=False,
            )
        )

    def get(
        self,
        i_perturbation: Optional[int] = None,
        train: bool = False,
        subset: bool = True,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        if subset == True and train == True:
            raise NotImplementedError("can only subset test data, not train data")
        if train:
            c48 = self._c48_train
            c384 = self._c384_train
        else:
            c48 = self._c48_test
            c384 = self._c384_test
        if i_perturbation is not None:
            c48 = c48.isel(perturbation=i_perturbation)
            c384 = c384.isel(perturbation=i_perturbation)
        if subset:
            c48 = c48.isel(time=self._c48_steps)
            c384 = c384.isel(time=self._c384_steps)
        return c48, c384

    @functools.lru_cache(maxsize=None)
    def get_transformed_c384(
        self,
        i_perturbation: Optional[int] = None,
        train: bool = False,
        subset: bool = True,
    ) -> xr.DataArray:
        c48, _ = self.get(i_perturbation, train, subset)
        array = self.transform(
            c48[self.varname].values.flatten().reshape(-1, 1)
        ).reshape(c48.shape)
        return xr.DataArray(
            array, dims=c48[self.varname].dims, coords=c48[self.varname].coords
        )

    @functools.lru_cache(maxsize=None)
    def get_predicted_c384(self, i_perturbation: Optional[int] = None) -> xr.DataArray:
        c48, _ = self.get(i_perturbation, train=False, subset=True)
        return self.cyclegan.predict(c48)[self.varname]


def evaluate(
    cyclegan: fv3fit.pytorch.CycleGAN,
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    varname: str,
):
    # c48_list = []
    # c384_list = []
    # for i in range(len(c384_real.perturbation)):
    #     c48_list.append(cyclegan.predict(c384_real.isel(perturbation=i), reverse=True))
    #     c384_list.append(cyclegan.predict(c48_real.isel(perturbation=i)))
    # c48_gen = xr.concat(c48_list, dim="perturbation")
    # c384_gen = xr.concat(c384_list, dim="perturbation")
    c48_gen = c48_real[cyclegan.state_variables]
    c384_gen = c384_real[cyclegan.state_variables]
    for varname in c48_gen.data_vars.keys():
        c48_real = c48_real.assign(
            **{
                varname: c48_real[varname] * c48_real[varname + "_std"]
                + c48_real[varname + "_mean"]
            }
        )
        c48_gen = c48_gen.assign(
            **{
                varname: c48_gen[varname] * c48_real[varname + "_std"]
                + c48_real[varname + "_mean"]
            }
        )
        c384_real = c384_real.assign(
            **{
                varname: c384_real[varname] * c384_real[varname + "_std"]
                + c384_real[varname + "_mean"]
            }
        )
        c384_gen = c384_gen.assign(
            **{
                varname: c384_gen[varname] * c384_real[varname + "_std"]
                + c384_real[varname + "_mean"]
            }
        )

    def plot_hist_all(varname, vmax=None):
        fig, ax = plt.subplots(
            len(c384_real.perturbation),
            2,
            figsize=(10, 1 + 2.5 * len(c384_real.perturbation)),
        )
        for i, climate in enumerate(c384_real.perturbation.values):
            plot_hist(i, varname, ax=ax[i, 0], log=False)
            plot_hist(i, varname, ax=ax[i, 1], log=True)
            ax[i, 0].set_title(f"{climate} (linear)\n" + ax[i, 0].get_title())
            ax[i, 1].set_title(f"{climate} (log)\n" + ax[i, 1].get_title())
            if vmax is not None:
                ax[i, 0].set_xlim(right=vmax)
                ax[i, 1].set_xlim(right=vmax)
        plt.tight_layout()
        if vmax is None:
            filename = "histogram.png"
        else:
            filename = f"histogram_vmax{vmax}.png"
        fig.savefig(filename, dpi=100)

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
        gen_cross_entropy = cross_entropy(
            c384_real[varname].isel(perturbation=i_perturbation).values.flatten(),
            c384_gen[varname].isel(perturbation=i_perturbation).values.flatten(),
        )
        c48_cross_entropy = cross_entropy(
            c384_real[varname].isel(perturbation=i_perturbation).values.flatten(),
            c48_real[varname].isel(perturbation=i_perturbation).values.flatten(),
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

    plot_hist_all(varname)
    # plot_hist_all(varname, vmax=5)

    c48_real_mean = c48_real.mean("time")
    c48_gen_mean = c48_gen.mean("time")
    c384_real_mean = c384_real.mean("time")
    c384_gen_mean = c384_gen.mean("time")

    def plot_mean_all(varname):
        fig, ax = plt.subplots(
            len(c384_real.perturbation),
            4,
            figsize=(18, 3 * len(c384_real.perturbation)),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        real = GRID.merge(c384_real_mean, compat="override")
        gen = GRID.merge(c384_gen_mean, compat="override")
        gen[f"{varname}_bias"] = gen[varname] - real[varname]
        gen[f"{varname}_c48_bias"] = c48_real_mean[varname] - real[varname]
        for i, climate in enumerate(c384_real.perturbation.values):
            vmin = min(
                c384_real_mean[varname].isel(perturbation=i).min().values,
                c384_gen_mean[varname].isel(perturbation=i).min().values,
            )
            vmax = max(
                c384_real_mean[varname].isel(perturbation=i).max().values,
                c384_gen_mean[varname].isel(perturbation=i).max().values,
            )
            bias_max = max(
                gen[f"{varname}_bias"].isel(perturbation=i).max().values,
                gen[f"{varname}_c48_bias"].isel(perturbation=i).max().values,
                -gen[f"{varname}_bias"].isel(perturbation=i).min().values,
                -gen[f"{varname}_c48_bias"].isel(perturbation=i).min().values,
            )
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
                vmin=-bias_max,
                vmax=bias_max,
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
                vmin=-bias_max,
                vmax=bias_max,
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


def train_quantile_mapping(c48_real: xr.Dataset, c384_real: xr.Dataset, varname: str):
    transform_c48 = sklearn.preprocessing.QuantileTransformer(
        subsample=int(1_000_000), output_distribution="uniform"
    )
    transform_c384 = sklearn.preprocessing.QuantileTransformer(
        subsample=int(1_000_000), output_distribution="uniform"
    )
    transform_c48.fit(c48_real[varname].values.flatten().reshape(-1, 1))
    transform_c384.fit(c384_real[varname].values.flatten().reshape(-1, 1))
    return QuantileMapping(transform_c48, transform_c384)


if __name__ == "__main__":
    fv3fit.set_random_seed(0)
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230130-231729-82b939d9-epoch_075/"  # precip-only
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230202-233100-c5d574a4-epoch_045/"  # precip-only, properly normalized
        "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230208-183103-cdda934c-epoch_017/"  # precip-only, properly normalized, +45 epochs
    ).to(DEVICE)
    VARNAME = "PRATEsfc"

    subselect_ratio = 1.0

    c384_real_all: xr.Dataset = (
        xr.open_zarr("./fine-combined.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c48_real_all: xr.Dataset = (
        xr.open_zarr("./coarse-combined.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )

    # c384_real: xr.Dataset = c384_real_all.isel(time=slice(2920, None))
    c384_real: xr.Dataset = c384_real_all.isel(time=slice(None, 2920))
    c384_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c384_real.time)),
            size=int(len(c384_real.time) * subselect_ratio),
            replace=False,
        )
    )
    # c384_steps=[0]
    c384_real = c384_real.isel(time=c384_steps)
    c48_real: xr.Dataset = c48_real_all.isel(time=slice(None, 11688))
    c48_steps = np.sort(
        np.random.choice(
            np.arange(0, len(c48_real.time)),
            size=int(len(c48_real.time) * subselect_ratio),
            replace=False,
        )
    )
    # c48_steps=[0]
    c48_real = c48_real.isel(time=c48_steps)

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
    evaluate(cyclegan, c48_real, c384_real, varname=VARNAME)

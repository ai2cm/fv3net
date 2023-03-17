# flake8: noqa
from typing import Tuple
import xarray as xr
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import fv3fit
from fv3fit.pytorch import DEVICE
import os
import cftime
from scipy.ndimage import uniform_filter1d
import scipy.optimize
import fv3viz
import cartopy.crs as ccrs
from vcm.catalog import catalog
import matplotlib.animation
import matplotlib.colors
import vcm
from fv3viz._plot_cube import UpdateablePColormesh


GRID = catalog["grid/c48"].read()
land_sea_mask = catalog["landseamask/c48"].read().land_sea_mask == 1

TO_MM_DAY = 86400 / 0.997


def convert_fine(ds_precip: xr.Dataset) -> xr.Dataset:
    # convert from 15-minute to 3-hourly
    assert timedelta(
        seconds=int((ds_precip.time[1] - ds_precip.time[0]).values.item() / 1e9)
    ) == timedelta(minutes=15)
    ds_precip = (
        ds_precip.coarsen(time=3 * 4, boundary="trim")
        .mean()
        .rename({"grid_xt_coarse": "grid_xt", "grid_yt_coarse": "grid_yt"})
    )
    precip = ds_precip["PRATEsfc_coarse"]
    return xr.Dataset({"PRATEsfc": precip, "TMPsfc": ds_precip["tsfc_coarse"]})


def get_sst_varying_mean_std(
    c48: xr.Dataset, c384: xr.Dataset, sst: np.ndarray
) -> Tuple[xr.DataArray, xr.DataArray]:
    i_sst = sst / 4.0 + 1.0
    i_sst[i_sst > 1.0] = 1 + (i_sst[i_sst > 1.0] - 1) / 2.0
    c48_mean = c48["PRATEsfc_mean"].isel(perturbation=[0, 1, 3])
    c48_std = c48["PRATEsfc_std"].isel(perturbation=[0, 1, 3])
    c384_mean = c384["PRATEsfc_mean"].isel(perturbation=[0, 1, 3])
    c384_std = c384["PRATEsfc_std"].isel(perturbation=[0, 1, 3])
    c48_varying_mean = np.zeros(sst.shape)
    c48_varying_std = np.zeros(sst.shape)
    c384_varying_mean = np.zeros(sst.shape)
    c384_varying_std = np.zeros(sst.shape)
    for i in range(0, 2):
        # linearly interpolate between successive mean/stds
        mask = i_sst.astype(int) == i
        if np.sum(mask) > 0:
            c48_varying_mean[mask] = c48_mean.isel(perturbation=i).values + (
                i_sst[mask] - i
            ) * (
                c48_mean.isel(perturbation=i + 1).values
                - c48_mean.isel(perturbation=i).values
            )
            c48_varying_std[mask] = c48_std.isel(perturbation=i).values + (
                i_sst[mask] - i
            ) * (
                c48_std.isel(perturbation=i + 1).values
                - c48_std.isel(perturbation=i).values
            )
            c384_varying_mean[mask] = c384_mean.isel(perturbation=i).values + (
                i_sst[mask] - i
            ) * (
                c384_mean.isel(perturbation=i + 1).values
                - c384_mean.isel(perturbation=i).values
            )
            c384_varying_std[mask] = c384_std.isel(perturbation=i).values + (
                i_sst[mask] - i
            ) * (
                c384_std.isel(perturbation=i + 1).values
                - c384_std.isel(perturbation=i).values
            )
    # there is no later value to interpolate to for the last value
    mask = i_sst.astype(int) == 2
    if np.sum(mask) > 0:
        c48_varying_mean[mask] = c48_mean.isel(perturbation=3).values
        c48_varying_std[mask] = c48_std.isel(perturbation=3).values
        c384_varying_mean[mask] = c384_mean.isel(perturbation=3).values
        c384_varying_std[mask] = c384_std.isel(perturbation=3).values

    # convert to xarray
    c48_varying_mean = xr.DataArray(
        c48_varying_mean, dims=["time"], name="PRATEsfc_mean",
    )
    c48_varying_std = xr.DataArray(c48_varying_std, dims=["time"], name="PRATEsfc_std",)
    c384_varying_mean = xr.DataArray(
        c384_varying_mean, dims=["time"], name="PRATEsfc_mean",
    )
    c384_varying_std = xr.DataArray(
        c384_varying_std, dims=["time"], name="PRATEsfc_std",
    )
    mean = xr.concat([c48_varying_mean, c384_varying_mean], dim="grid").assign_coords(
        {"grid": ["C48", "C384"]}
    )
    std = xr.concat([c48_varying_std, c384_varying_std], dim="grid").assign_coords(
        {"grid": ["C48", "C384"]}
    )
    return mean, std


def get_sst_offsets(time: xr.DataArray):
    interpolation_knots = {
        cftime.DatetimeJulian(2017, 7, 15): 0.0,
        cftime.DatetimeJulian(2017, 11, 1): 0.0,
        cftime.DatetimeJulian(2018, 11, 1): 1.0,
        cftime.DatetimeJulian(2019, 11, 1): 2.0,
        cftime.DatetimeJulian(2020, 11, 1): 3.0,
        cftime.DatetimeJulian(2021, 11, 1): 4.0,
        cftime.DatetimeJulian(2022, 11, 1): 5.0,
    }
    offsets = xr.DataArray(
        list(interpolation_knots.values()),
        dims=["time"],
        coords=[list(interpolation_knots.keys())],
    )
    interpolated_offsets = offsets.interp(time=time)
    return interpolated_offsets


def exponential_cross_entropy(p, q, bins):
    """
    Compute cross-entropy of two probability distributions after fitting
    them to exponential distributions.
    """
    p_lambda = fit_exponential(p, bins)
    q_lambda = fit_exponential(q, bins)


# def fit_exponential(p, bins, n_exponentials: int):
#     """
#     Fit a probability distribution to an exponential distribution.
#     """
#     # p_norm = p / np.sum(p)
#     # p_cdf = np.concatenate([0, np.cumsum(p_norm)])

#     p_norm = p / np.sum(p)
#     p_cdf = np.concatenate([[0], np.cumsum(p_norm)])
#     # use least squares fitting against exponential CDF to find a, b
#     # that minimize the error

#     # the exponential CDF is given by 1 - exp(-lambda * x)

#     def error(params):
#         coeffs, lambdas = params[:n_exponentials], params[n_exponentials:]
#         coeffs = np.abs(coeffs)
#         lambdas = np.abs(lambdas)
#         coeffs = coeffs / np.sum(coeffs)
#         residual = np.copy(p_cdf)
#         for i in range(n_exponentials):
#             residual -= coeffs[i] * (1 - np.exp(-lambdas[i] * bins))
#         # return np.sum((p_cdf - c*(1 - np.exp(-a * bins)) - (1 - c)*(1 - np.exp(-b * bins))) ** 2)
#         return np.sum(residual ** 2) + 1e-8 * np.sum((coeffs - 1. / n_exponentials) ** 2)

#     # the mean is a good initial guess for lambda
#     mean = np.sum(p_norm * bins[1:])
#     initial_coeffs = [1.0 / n_exponentials for _ in range(n_exponentials)]
#     initial_lambdas = [1.0 / mean*(1-0.001*i) for i in range(n_exponentials)]
#     result = scipy.optimize.minimize(error, initial_coeffs + initial_lambdas).x
#     coeffs, lambdas = result[:n_exponentials], result[n_exponentials:]
#     coeffs = np.abs(coeffs)
#     lambdas = np.abs(lambdas)
#     sum_coeffs = np.sum(coeffs)
#     for i, coeff in enumerate(coeffs):
#         coeffs[i] = coeff / sum_coeffs
#     # a, b, c = scipy.optimize.minimize(error, [1.0 / mean, 0.99*mean, 0.5]).x

#     # plot the CDF and PDF
#     # CDF
#     cdf_fit = np.zeros_like(p_cdf)
#     for i in range(n_exponentials):
#         cdf_fit += coeffs[i] * (1 - np.exp(-lambdas[i] * bins))
#     plt.figure()
#     plt.plot(bins, p_cdf, label="data")
#     plt.plot(bins, cdf_fit, label="fit")
#     plt.legend()
#     plt.savefig(f"sst-cdf-{n_exponentials}.png")
#     # PDF
#     mid_bin = 0.5 * (bins[1:] + bins[:-1])
#     pdf_fit = np.zeros_like(mid_bin)
#     for i in range(n_exponentials):
#         pdf_fit += coeffs[i] * lambdas[i] * np.exp(-lambdas[i] * mid_bin)
#     plt.figure()
#     plt.plot(mid_bin, p_norm, label="data")
#     plt.plot(mid_bin, pdf_fit * np.diff(bins), label="fit")
#     plt.yscale("log")
#     plt.legend()
#     plt.savefig(f"sst-pdf-{n_exponentials}.png")
#     print(coeffs, lambdas)
#     return coeffs, lambdas


def fit_exponential(p, bins, label):
    """
    Fit a probability distribution to an exponential distribution.
    """
    # p_norm = p / np.sum(p)
    # p_cdf = np.concatenate([0, np.cumsum(p_norm)])

    p_norm = p / np.sum(p)
    p_cdf = 1.0 - np.concatenate([[0], np.cumsum(p_norm)])
    i_start = np.argmax(p_cdf < 0.01)
    i_end = np.argmin(p_norm)
    p_cdf = p_cdf[i_start : i_end + 1]
    bins = bins[i_start : i_end + 1]
    p_norm = p_norm[i_start:i_end]
    print(i_start, i_end)
    # use least squares fitting against exponential CDF to find a, b
    # that minimize the error

    # the exponential CDF is given by 1 - exp(-lambda * x)

    def error(params):
        a = params
        return np.sum((p_cdf - (1 - np.exp(-a * bins))) ** 2)

    # the mean is a good initial guess for lambda
    mean = np.sum(p_norm * bins[1:])
    print(1.0 / mean)
    a = scipy.optimize.minimize(error, [1.0 / mean]).x

    # plot the CDF and PDF
    # CDF
    plt.figure()
    plt.plot(bins, p_cdf, label="data")
    # plt.plot(bins[i_start], p_cdf[i_start], "o", label="99th pct")
    # plt.plot(bins[i_end-1], p_cdf[i_end-1], "o", label="1st zero")
    # plt.plot(bins, np.exp(-a * bins), label="fit, lambda={:.2f}".format(float(a)))
    # plt.ylim(0.999*p_cdf[i_start], 1.0)
    plt.title(label)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"sst-cdf-{label}.png")
    # PDF
    mid_bin = 0.5 * (bins[1:] + bins[:-1])
    plt.figure()
    plt.plot(mid_bin, p_norm, label="data")
    # plt.plot(mid_bin[i_start], p_norm[i_start], "o", label="99th pct")
    # plt.plot(mid_bin[i_end-2], p_norm[i_end-2], "o", label="1st zero")
    plt.plot(
        mid_bin,
        a * np.exp(-a * mid_bin) * np.diff(bins),
        label="fit, lambda={:.2f}".format(float(a)),
    )
    plt.title(label)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"sst-pdf-{label}.png")
    return a


def plot_mean_all(c48, c384, c48_gen, c384_gen, label: str):
    varname = "PRATEsfc"
    fig, ax = plt.subplots(
        2, 4, figsize=(16, 6), subplot_kw={"projection": ccrs.Robinson()},
    )
    precip_c384 = c384[varname].mean("time") * TO_MM_DAY
    precip_c48 = c48[varname].mean("time") * TO_MM_DAY
    precip_c384_gen = c384_gen[varname].mean("time") * TO_MM_DAY
    precip_c48_gen = c48_gen[varname].mean("time") * TO_MM_DAY

    vmin = min(
        precip_c384.min().values,
        precip_c48.min().values,
        precip_c384_gen.min().values,
        precip_c48_gen.min().values,
    )
    vmax = max(
        precip_c384.max().values,
        precip_c48.max().values,
        precip_c384_gen.max().values,
        precip_c48_gen.max().values,
    )

    ax[0, 0].set_title("c384_real")
    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({varname: precip_c384}), compat="override"),
        var_name=varname,
        ax=ax[0, 0],
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 1].set_title("c384_gen")
    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({varname: precip_c384_gen}), compat="override"),
        var_name=varname,
        ax=ax[0, 1],
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 2].set_title("c48_real")
    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({varname: precip_c48}), compat="override"),
        var_name=varname,
        ax=ax[0, 2],
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 3].set_title("c48_gen")
    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({varname: precip_c48_gen}), compat="override"),
        var_name=varname,
        ax=ax[0, 3],
        vmin=vmin,
        vmax=vmax,
    )
    gen_bias = precip_c384_gen - precip_c384
    c48_bias = precip_c48 - precip_c384
    c48_bias_mean = c48_bias.mean().values
    c48_bias_std = c48_bias.std().values
    c48_bias_land_std = c48_bias.where(land_sea_mask).std().values
    c48_bias_land_mean = c48_bias.where(land_sea_mask).mean().values
    gen_bias_mean = gen_bias.mean().values
    gen_bias_std = gen_bias.std().values
    gen_bias_land_std = gen_bias.where(land_sea_mask).std().values
    gen_bias_land_mean = gen_bias.where(land_sea_mask).mean().values

    bias_min = min(gen_bias.min().values, c48_bias.min().values)
    bias_max = max(gen_bias.max().values, c48_bias.max().values)
    bias_max = max(abs(bias_min), abs(bias_max))
    bias_min = -bias_max

    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({f"{varname}_gen_bias": gen_bias}), compat="override"),
        var_name=f"{varname}_gen_bias",
        ax=ax[1, 1],
        vmin=bias_min,
        vmax=bias_max,
    )
    ax[1, 1].set_title(
        "gen_bias\nmean: {:.2e}\nstd: {:.2e}\nland mean:{:.2e}\nland std: {:.2e}".format(
            gen_bias_mean, gen_bias_std, gen_bias_land_mean, gen_bias_land_std
        )
    )
    fv3viz.plot_cube(
        ds=GRID.merge(xr.Dataset({f"{varname}_c48_bias": c48_bias}), compat="override"),
        var_name=f"{varname}_c48_bias",
        ax=ax[1, 2],
        vmin=bias_min,
        vmax=bias_max,
    )
    ax[1, 2].set_title(
        "c48_bias\nmean: {:.2e}\nstd: {:.2e}\nland mean:{:.2e}\nland std: {:.2e}".format(
            c48_bias_mean, c48_bias_std, c48_bias_land_mean, c48_bias_land_std
        )
    )

    plt.tight_layout()
    fig.savefig(f"ramping-mean-{label}.png", dpi=100)


def to_diurnal_land(ds: xr.Dataset):
    local_time = vcm.local_time(
        GRID.merge(ds, compat="override"), time="time", lon_var="lon"
    )
    return (
        ds.where(land_sea_mask)
        .groupby_bins(local_time, bins=np.arange(0, 25, 3))
        .mean()
    )


def plot_diurnal_land(c48, c384, c48_gen, c384_gen):
    c48_gen = c48_gen.update({"time": c384.time})
    c384_gen = c384_gen.update({"time": c48.time})
    c48 = to_diurnal_land(c48)
    c384 = to_diurnal_land(c384)
    c48_gen = to_diurnal_land(c48_gen)
    c384_gen = to_diurnal_land(c384_gen)
    # data = ds[f"{varname}_mean"].mean(dim=("tile", "x", "y")) #.groupby(in_region).mean().sel(group=1)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),)
    x = np.arange(0, 24, 3)
    ax.plot(x, c48["PRATEsfc"].values * TO_MM_DAY, label="c48")
    ax.plot(x, c384["PRATEsfc"].values * TO_MM_DAY, label="c384")
    ax.plot(x, c48_gen["PRATEsfc"].values * TO_MM_DAY, label="c48_gen")
    ax.plot(x, c384_gen["PRATEsfc"].values * TO_MM_DAY, label="c384_gen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}:00" for i in range(0, 24, 3)])
    ax.set_xlabel("local time")
    ax.set_ylabel("mm/day")
    ax.set_title("PRATEsfc over land")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"ramping-diurnal_cycle.png", dpi=100)


def animate_all(c48, c384, c48_gen, c384_gen, label: str):

    fig, ax = plt.subplots(
        2, 2, figsize=(12, 6), subplot_kw={"projection": ccrs.Robinson()},
    )
    varname = "PRATEsfc"
    precip_c384 = c384[varname].transpose("tile", "time", "y", "x") * TO_MM_DAY
    precip_c48 = c48[varname].transpose("tile", "time", "y", "x") * TO_MM_DAY
    precip_c384_gen = c384_gen[varname].transpose("tile", "time", "y", "x") * TO_MM_DAY
    precip_c48_gen = c48_gen[varname].transpose("tile", "time", "y", "x") * TO_MM_DAY

    # im_c384 = UpdateablePColormesh(lat, lon, array, ax=ax[0, 0])
    vmin = min(
        precip_c384.min().values,
        precip_c48.min().values,
        precip_c384_gen.min().values,
        precip_c48_gen.min().values,
    )
    vmax = max(
        precip_c384.max().values,
        precip_c48.max().values,
        precip_c384_gen.max().values,
        precip_c48_gen.max().values,
    )

    nt = min(
        len(precip_c384.time),
        len(precip_c48.time),
        len(precip_c384_gen.time),
        len(precip_c48_gen.time),
    )
    # nt = 500
    print(f"animating {nt} frames")

    lat = GRID["latb"].values
    lon = GRID["lonb"].values
    norm = matplotlib.colors.SymLogNorm(
        linthresh=1, linscale=1, vmin=0, vmax=vmax, base=10
    )
    im_c384 = UpdateablePColormesh(
        lat,
        lon,
        precip_c48.isel(time=0).values,
        ax=ax[0, 0],
        vmin=0,
        vmax=vmax,
        norm=norm,
    )
    ax[0, 0].set_title("c48_real")
    im_c384_gen = UpdateablePColormesh(
        lat,
        lon,
        precip_c48_gen.isel(time=0).values,
        ax=ax[0, 1],
        vmin=0,
        vmax=vmax,
        norm=norm,
    )
    ax[0, 1].set_title("c48_gen")
    im_c48 = UpdateablePColormesh(
        lat,
        lon,
        precip_c384.isel(time=0).values,
        ax=ax[1, 0],
        vmin=0,
        vmax=vmax,
        norm=norm,
    )
    ax[1, 0].set_title("c384_real")
    im_c48_gen = UpdateablePColormesh(
        lat,
        lon,
        precip_c384_gen.isel(time=0).values,
        ax=ax[1, 1],
        vmin=0,
        vmax=vmax,
        norm=norm,
    )
    ax[1, 1].set_title("c384_gen")
    for i in range(2):
        for j in range(2):
            ax[i, j].coastlines()
    fig.suptitle(f"{varname} days = 0.00")

    def func(data):
        im_c384.update(data[0])
        im_c384_gen.update(data[1])
        im_c48.update(data[2])
        im_c48_gen.update(data[3])
        i = data[4]
        days = i
        fig.suptitle(f"{varname} days = {days:.2f}")

    frames = (
        (
            precip_c48.isel(time=i).values,
            precip_c48_gen.isel(time=i).values,
            precip_c384.isel(time=i).values,
            precip_c384_gen.isel(time=i).values,
            i,
        )
        for i in range(0, nt)
    )
    ani = matplotlib.animation.FuncAnimation(
        fig, func, frames=frames, interval=100, repeat=False, save_count=nt,
    )
    ani.save(
        f"basic_animation_{nt}-{label}.mp4", fps=10, extra_args=["-vcodec", "libx264"]
    )


if __name__ == "__main__":
    fv3fit.set_random_seed(0)
    CHECKPOINT_PATH = "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/"
    BASE_NAME, EPOCH = (
        # "20230208-183103-cdda934c", 17  # precip-only, properly normalized, +45 epochs
        # "20230302-000015-699b0906", 75  # "no-demean-2e-6-e75"
        "20230314-214027-54366191",
        17
        # "20230303-203306-f753d490", 69  # "denorm-2e-6-3x3-e69"
    )
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230130-231729-82b939d9-epoch_075/"  # precip-only
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230202-233100-c5d574a4-epoch_045/"  # precip-only, properly normalized
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230208-183103-cdda934c-epoch_017/"  # precip-only, properly normalized, +45 epochs
        CHECKPOINT_PATH
        + BASE_NAME
        + f"-epoch_{EPOCH:03d}/"
    ).to(DEVICE)
    VARNAME = "PRATEsfc"

    if not os.path.exists("c48-ramping.zarr"):
        # c48: xr.Dataset = xr.open_zarr("gs://vcm-ml-experiments/spencerc/2022-07-07/n2f-25km-baseline-increasing-sst/fv3gfs_run/sfc_dt_atmos.zarr")[["PRATEsfc", "TMPsfc"]]
        c48 = (
            xr.open_zarr(
                "gs://vcm-ml-experiments/spencerc/2022-07-07/n2f-25km-baseline-increasing-sst/fv3gfs_run/diags.zarr"
            )
            .coarsen(time=3)
            .mean()
            .rename({"total_precipitation_rate": "PRATEsfc"})[["PRATEsfc"]]
        )
        c48.rename({"x": "grid_xt", "y": "grid_yt"}).chunk(
            {"time": 80, "tile": 6, "grid_xt": 48, "grid_yt": 48}
        ).to_zarr("c48-ramping.zarr")
    c48 = xr.open_zarr("c48-ramping.zarr")[["PRATEsfc"]]
    if not os.path.exists("c384-ramping.zarr"):
        c384 = convert_fine(
            xr.open_zarr(
                "gs://vcm-ml-raw-flexible-retention/2022-07-29-increasing-SST-C384-FV3GFS-run/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr"
            )
        )[["PRATEsfc"]]
        c384.chunk({"time": 365, "tile": 6, "grid_xt": 48, "grid_yt": 48}).to_zarr(
            "c384-ramping.zarr"
        )
    c384 = xr.open_zarr("c384-ramping.zarr")[["PRATEsfc"]]

    c48 = c48.rename({"grid_xt": "x", "grid_yt": "y"})
    c384 = c384.rename({"grid_xt": "x", "grid_yt": "y"})

    # c48 = c48.rename({"grid_xt": "x", "grid_yt": "y"})
    # c384 = c384.rename({"grid_xt": "x", "grid_yt": "y"})

    c48_norm_data = xr.open_zarr("coarse-combined-no-demean.zarr")
    c384_norm_data = xr.open_zarr("fine-combined-no-demean.zarr")
    sst_c48 = get_sst_offsets(c48.time).values
    sst_c384 = get_sst_offsets(c384.time).values
    mean_c48, std_c48 = get_sst_varying_mean_std(c48_norm_data, c384_norm_data, sst_c48)
    mean_c384, std_c384 = get_sst_varying_mean_std(
        c48_norm_data, c384_norm_data, sst_c384
    )

    # HACK: for denorm case
    mean_c48[:] = 0.0
    std_c48[:] = 1.0
    mean_c384[:] = 0.0
    std_c384[:] = 1.0

    plt.figure()
    plt.plot(sst_c48, mean_c48.sel(grid="C48").values, label="C48")
    plt.plot(sst_c384, mean_c384.sel(grid="C384").values, label="C384")
    plt.plot([-4, 0, 4, 8], c48_norm_data["PRATEsfc_mean"].values, "o", label="C48_ref")
    plt.plot(
        [-4, 0, 4, 8], c384_norm_data["PRATEsfc_mean"].values, "o", label="C384_ref"
    )
    plt.legend()
    plt.title("Mean")
    plt.tight_layout()
    plt.savefig("sst-mean.png")
    plt.figure()
    plt.plot(sst_c48, std_c48.sel(grid="C48").values, label="C48")
    plt.plot(sst_c384, std_c384.sel(grid="C384").values, label="C384")
    plt.plot([-4, 0, 4, 8], c48_norm_data["PRATEsfc_std"].values, "o", label="C48_ref")
    plt.plot(
        [-4, 0, 4, 8], c384_norm_data["PRATEsfc_std"].values, "o", label="C384_ref"
    )
    plt.legend()
    plt.title("Std")
    plt.tight_layout()
    plt.savefig("sst-std.png")

    FILENAME = f"predicted-ramping-{BASE_NAME}" + "-{res}.nc"
    if not os.path.exists(FILENAME.format(res="c48")):
        c384_norm = (c384 - mean_c384.sel(grid="C384")) / std_c384.sel(grid="C384")
        c48_norm_gen = cyclegan.predict(c384_norm, reverse=True)
        c48_gen = c48_norm_gen * std_c384.sel(grid="C48") + mean_c384.sel(grid="C48")
        c48_gen.to_netcdf(FILENAME.format(res="c48"))
    if not os.path.exists(FILENAME.format(res="c384")):
        c48_norm = (c48 - mean_c48.sel(grid="C48")) / std_c48.sel(grid="C48")
        c384_norm_gen = cyclegan.predict(c48_norm)
        c384_gen = c384_norm_gen * std_c48.sel(grid="C384") + mean_c48.sel(grid="C384")
        c384_gen.to_netcdf(FILENAME.format(res="c384"))
    c48_gen = xr.open_dataset(FILENAME.format(res="c48"))
    c384_gen = xr.open_dataset(FILENAME.format(res="c384"))

    # plot_diurnal_land(c48, c384, c48_gen, c384_gen)

    c48 = c48.coarsen(time=8).mean()
    c384 = c384.coarsen(time=8).mean()
    c48_gen = c48_gen.coarsen(time=8).mean()
    c384_gen = c384_gen.coarsen(time=8).mean()

    c48 = c48.isel(time=slice(2 * 365, 3 * 365))
    c384 = c384.isel(time=slice(2 * 365, 3 * 365))
    c48_gen = c48_gen.isel(time=slice(2 * 365, 3 * 365))
    c384_gen = c384_gen.isel(time=slice(2 * 365, 3 * 365))

    animate_all(c48, c384, c48_gen, c384_gen, f"{BASE_NAME}-e{EPOCH}")

    plot_mean_all(c48, c384, c48_gen, c384_gen, f"{BASE_NAME}-e{EPOCH}")

    n_smoothing = 30
    plt.figure()
    plt.plot(
        c48.time.values,
        uniform_filter1d(
            c48[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C48",
    )
    plt.plot(
        c384.time.values,
        uniform_filter1d(
            c48_gen[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C48 gen",
    )
    plt.plot(
        c384.time.values,
        uniform_filter1d(
            c384[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C384",
    )
    plt.plot(
        c48.time.values,
        uniform_filter1d(
            c384_gen[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C384 gen",
    )
    # plt.plot(c384.time.values, c48_gen[VARNAME].mean(dim=("tile", "x", "y")).values, label="C48 gen")
    # plt.plot(c384.time.values, c384[VARNAME].mean(dim=("tile", "x", "y")).values, label="C384")
    # plt.plot(c48.time.values, c384_gen[VARNAME].mean(dim=("tile", "x", "y")).values, label="C384 gen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ramping-{BASE_NAME}-e{EPOCH}.png", dpi=100)

    n_bins = 100
    bins = np.linspace(0, 1e-2, 101)
    v1 = 1e-2 / n_bins ** 2
    np.concatenate(
        [[-1.0, 0.0], np.logspace(np.log10(v1), np.log10(1e-2), n_bins - 1),]
    )
    time_bin = 30
    c48_hist = np.zeros((len(c48.time) // time_bin, len(bins) - 1))
    c48_gen_hist = np.zeros((len(c384.time) // (time_bin), len(bins) - 1))
    c384_hist = np.zeros((len(c384.time) // (time_bin), len(bins) - 1))
    c384_gen_hist = np.zeros((len(c48.time) // time_bin, len(bins) - 1))
    for i_hist, i in enumerate(range(0, len(c48.time) - time_bin, time_bin)):
        c48_hist[i_hist, :] = np.histogram(
            c48[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
            bins=bins,
            density=True,
        )[0]
        c384_gen_hist[i_hist, :] = np.histogram(
            c384_gen[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
            bins=bins,
            density=True,
        )[0]
    for i_hist, i in enumerate(range(0, len(c384.time) - time_bin, time_bin)):
        c384_hist[i_hist, :] = np.histogram(
            c384[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
            bins=bins,
            density=True,
        )[0]
        c48_gen_hist[i_hist, :] = np.histogram(
            c48_gen[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
            bins=bins,
            density=True,
        )[0]

    fit_exponential(np.sum(c48_hist, axis=0), bins * TO_MM_DAY, "c48")
    fit_exponential(np.sum(c384_hist, axis=0), bins * TO_MM_DAY, "c384")
    fit_exponential(np.sum(c48_gen_hist, axis=0), bins * TO_MM_DAY, "c48_gen")
    fit_exponential(np.sum(c384_gen_hist, axis=0), bins * TO_MM_DAY, "c384_gen")

    # # for i, t in enumerate(c48.time.values):
    # #     c48_hist[i, :] = np.histogram(c48[VARNAME].sel(time=t).values, bins=bins, density=True)[0]
    # #     c48_gen_hist[i, :] = np.histogram(c48_gen[VARNAME].sel(time=t).values, bins=bins, density=True)[0]
    # # for i, t in enumerate(c384.time.values):
    # #     c384_hist[i, :] = np.histogram(c384[VARNAME].sel(time=t).values, bins=bins, density=True)[0]
    # #     c384_gen_hist[i, :] = np.histogram(c384_gen[VARNAME].sel(time=t).values, bins=bins, density=True)[0]
    # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # im = ax[0, 0].pcolormesh(c48.time.values[::time_bin], bins[:-1] * TO_MM_DAY, c48_hist.T, norm=colors.LogNorm(vmin=1e-2, vmax=c48_hist.max()))
    # plt.colorbar(im, ax=ax[0, 0])
    # im = ax[0, 1].pcolormesh(c384.time.values[::time_bin], bins[:-1] * TO_MM_DAY, c48_gen_hist.T, norm=colors.LogNorm(vmin=1e-2, vmax=c48_gen_hist.max()))
    # plt.colorbar(im, ax=ax[0, 1])
    # im = ax[1, 0].pcolormesh(c384.time.values[::time_bin], bins[:-1] * TO_MM_DAY, c384_hist.T, norm=colors.LogNorm(vmin=1e-2, vmax=c384_hist.max()))
    # plt.colorbar(im, ax=ax[1, 0])
    # im = ax[1, 1].pcolormesh(c48.time.values[::time_bin], bins[:-1] * TO_MM_DAY, c384_gen_hist.T, norm=colors.LogNorm(vmin=1e-2, vmax=c384_gen_hist.max()))
    # plt.colorbar(im, ax=ax[1, 1])
    # ax[0, 0].set_title("C48")
    # ax[0, 1].set_title("C48 gen")
    # ax[1, 0].set_title("C384")
    # ax[1, 1].set_title("C384 gen")
    # plt.tight_layout()
    # plt.savefig("sst-hist.png", dpi=100)

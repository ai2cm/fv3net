# flake8: noqa
import shutil
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
from process_combined_aggregate import (
    plot_diurnal_cycle,
    plot_mean_all,
    plot_cdf_dual_pane,
    DatasetAggregator,
)


GRID = catalog["grid/c48"].read()
land_sea_mask = catalog["landseamask/c48"].read().land_sea_mask == 1

TO_MM_DAY = 86400 / 0.997


def convert_fine(ds_precip: xr.Dataset) -> xr.Dataset:
    # data must be 3-hourly
    assert timedelta(
        seconds=int((ds_precip.time[1] - ds_precip.time[0]).values.item() / 1e9)
    ) == timedelta(hours=3)
    ds_precip = ds_precip.rename(
        {"grid_xt_coarse": "grid_xt", "grid_yt_coarse": "grid_yt"}
    )
    precip = ds_precip["PRATEsfc_coarse"]
    return xr.Dataset({"PRATEsfc": precip})


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


# def plot_mean_all(c48, c384, c48_gen, c384_gen, label: str):
#     varname = "PRATEsfc"
#     fig, ax = plt.subplots(
#         2, 4, figsize=(16, 6), subplot_kw={"projection": ccrs.Robinson()},
#     )
#     precip_c384 = c384[varname].mean("time") * TO_MM_DAY
#     precip_c48 = c48[varname].mean("time") * TO_MM_DAY
#     precip_c384_gen = c384_gen[varname].mean("time") * TO_MM_DAY
#     precip_c48_gen = c48_gen[varname].mean("time") * TO_MM_DAY

#     vmin = min(
#         precip_c384.min().values,
#         precip_c48.min().values,
#         precip_c384_gen.min().values,
#         precip_c48_gen.min().values,
#     )
#     vmax = max(
#         precip_c384.max().values,
#         precip_c48.max().values,
#         precip_c384_gen.max().values,
#         precip_c48_gen.max().values,
#     )

#     ax[0, 0].set_title("c384_real")
#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({varname: precip_c384}), compat="override"),
#         var_name=varname,
#         ax=ax[0, 0],
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax[0, 1].set_title("c384_gen")
#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({varname: precip_c384_gen}), compat="override"),
#         var_name=varname,
#         ax=ax[0, 1],
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax[0, 2].set_title("c48_real")
#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({varname: precip_c48}), compat="override"),
#         var_name=varname,
#         ax=ax[0, 2],
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax[0, 3].set_title("c48_gen")
#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({varname: precip_c48_gen}), compat="override"),
#         var_name=varname,
#         ax=ax[0, 3],
#         vmin=vmin,
#         vmax=vmax,
#     )
#     gen_bias = precip_c384_gen - precip_c384
#     c48_bias = precip_c48 - precip_c384
#     c48_bias_mean = c48_bias.mean().values
#     c48_bias_std = c48_bias.std().values
#     c48_bias_land_std = c48_bias.where(land_sea_mask).std().values
#     c48_bias_land_mean = c48_bias.where(land_sea_mask).mean().values
#     gen_bias_mean = gen_bias.mean().values
#     gen_bias_std = gen_bias.std().values
#     gen_bias_land_std = gen_bias.where(land_sea_mask).std().values
#     gen_bias_land_mean = gen_bias.where(land_sea_mask).mean().values

#     bias_min = min(gen_bias.min().values, c48_bias.min().values)
#     bias_max = max(gen_bias.max().values, c48_bias.max().values)
#     bias_max = max(abs(bias_min), abs(bias_max))
#     bias_min = -bias_max

#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({f"{varname}_gen_bias": gen_bias}), compat="override"),
#         var_name=f"{varname}_gen_bias",
#         ax=ax[1, 1],
#         vmin=bias_min,
#         vmax=bias_max,
#     )
#     ax[1, 1].set_title(
#         "gen_bias\nmean: {:.2e}\nstd: {:.2e}\nland mean:{:.2e}\nland std: {:.2e}".format(
#             gen_bias_mean, gen_bias_std, gen_bias_land_mean, gen_bias_land_std
#         )
#     )
#     fv3viz.plot_cube(
#         ds=GRID.merge(xr.Dataset({f"{varname}_c48_bias": c48_bias}), compat="override"),
#         var_name=f"{varname}_c48_bias",
#         ax=ax[1, 2],
#         vmin=bias_min,
#         vmax=bias_max,
#     )
#     ax[1, 2].set_title(
#         "c48_bias\nmean: {:.2e}\nstd: {:.2e}\nland mean:{:.2e}\nland std: {:.2e}".format(
#             c48_bias_mean, c48_bias_std, c48_bias_land_mean, c48_bias_land_std
#         )
#     )

#     plt.tight_layout()
#     fig.savefig(f"plots/ramping-mean-{label}.png", dpi=100)


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
        2, 2, figsize=(12, 7), subplot_kw={"projection": ccrs.Robinson()},
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
        plot_colorbar=False,
    )
    ax[0, 0].set_title("C48")
    im_c384_gen = UpdateablePColormesh(
        lat,
        lon,
        precip_c48_gen.isel(time=0).values,
        ax=ax[0, 1],
        vmin=0,
        vmax=vmax,
        norm=norm,
        plot_colorbar=False,
    )
    ax[0, 1].set_title("C48 (ML)")
    im_c48 = UpdateablePColormesh(
        lat,
        lon,
        precip_c384.isel(time=0).values,
        ax=ax[1, 0],
        vmin=0,
        vmax=vmax,
        norm=norm,
        plot_colorbar=False,
    )
    ax[1, 0].set_title("C384")
    im_c48_gen = UpdateablePColormesh(
        lat,
        lon,
        precip_c384_gen.isel(time=0).values,
        ax=ax[1, 1],
        vmin=0,
        vmax=vmax,
        norm=norm,
        plot_colorbar=False,
    )
    ax[1, 1].set_title("C384 (ML)")
    for i in range(2):
        for j in range(2):
            ax[i, j].coastlines()
    fig.suptitle(f"Surface precipitation (mm/day), elapsed days = 0.00")

    def func(data):
        im_c384.update(data[0])
        im_c384_gen.update(data[1])
        im_c48.update(data[2])
        im_c48_gen.update(data[3])
        i = data[4]
        days = i / 8.0
        fig.suptitle(f"Surface precipitation (mm/day), elapsed days = {days:.2f}")

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
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im_c48_gen.handles[-1], cax=cbar_ax)
    plt.tight_layout(rect=[0.03, 0.03, 0.89, 0.93])
    ani = matplotlib.animation.FuncAnimation(
        fig, func, frames=frames, interval=100, repeat=False, save_count=nt,
    )
    ani.save(
        f"basic_animation_{nt}-{label}.mp4", fps=10, extra_args=["-vcodec", "libx264"]
    )


def plot_annual_means(
    c48_real_all: xr.Dataset,
    c384_real_all: xr.Dataset,
    c48_gen_all: xr.Dataset,
    c384_gen_all: xr.Dataset,
    label: str,
):
    ds_list = []
    for i_year in range(4):
        aggregator = DatasetAggregator()
        aggregator.add(
            c48_real_all.transpose("tile", "time", "x", "y")
            .expand_dims("perturbation")
            .isel(time=slice(i_year * 365 * 8, (i_year + 1) * 365 * 8)),
            c384_real_all.transpose("tile", "time", "x", "y")
            .expand_dims("perturbation")
            .isel(time=slice(i_year * 365 * 8, (i_year + 1) * 365 * 8)),
            c48_gen_all.expand_dims("perturbation").isel(
                time=slice(i_year * 365 * 8, (i_year + 1) * 365 * 8)
            ),
            c384_gen_all.expand_dims("perturbation").isel(
                time=slice(i_year * 365 * 8, (i_year + 1) * 365 * 8)
            ),
        )
        ds = aggregator.get_dataset()
        ds = ds.assign_coords({"perturbation": [f"Year {i_year}"]})
        ds_list.append(ds)
    ds = xr.concat(ds_list, dim="perturbation")
    plot_mean_all(ds, "PRATEsfc", f"ramping-{label}")


def plot_weather_evaluation(c48_real, c384_real, c48_gen, c384_gen, label: str):
    c48_rmse = (c384_real - c48_real).std(dim=["tile", "x", "y"])["PRATEsfc"].values
    c384_gen_rmse = (
        (c384_gen - c384_real).std(dim=["tile", "x", "y"])["PRATEsfc"].values
    )
    dt = 1.0 / 8
    time = np.arange(dt, (len(c48_rmse) + 1) * dt, dt)
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(time, c48_rmse * TO_MM_DAY, label="c48")
    ax.plot(time, c384_gen_rmse * TO_MM_DAY, label="c384_gen")
    ax.legend()
    ax.set_xlabel("days elapsed")
    ax.set_ylabel("Precipitation RMSE (mm/day)")
    ax.set_title(f"Weather evaluation")
    plt.tight_layout()
    plt.savefig(f"ramping-rmse-{label}.png")


if __name__ == "__main__":
    fv3fit.set_random_seed(0)
    CHECKPOINT_PATH = "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/"
    BASE_NAME, EPOCH = (
        # "20230208-183103-cdda934c", 17  # precip-only, properly normalized, +45 epochs
        # "20230302-000015-699b0906", 75  # "no-demean-2e-6-e75"
        # "20230314-214027-54366191", 17
        # "20230316-151658-9017348e",
        # 35  # lr-1e-4-decay-0.79433-no-geo-bias
        # "20230303-203306-f753d490", 69  # "denorm-2e-6-3x3-e69"
        "20230329-221949-9d8e8abc",
        16,  # "prec-lr-1e-4-decay-0.63096-full",
        # "20230314-213709-fc95b736", 15 # "lr-1e-4-decay-0.79433"
        # "20230424-183552-125621da", 16,  # "prec-lr-1e-4-decay-0.63096-full-no-geo-features"
        # "20230424-191937-253268fd", 16,  # "prec-lr-1e-4-decay-0.63096-full-no-identity-loss"
        # "20230427-160655-b8b010ce", 16,  # "prec-lr-1e-4-decay-0.63096-1-year"
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
    LOCAL_C48_FILENAME = "./ramping_data/c48-ramping-full.zarr"
    LOCAL_C384_FILENAME = "./ramping_data/c384-ramping-full.zarr"

    if not os.path.exists(LOCAL_C48_FILENAME):
        # c48: xr.Dataset = xr.open_zarr("gs://vcm-ml-experiments/spencerc/2022-07-07/n2f-25km-baseline-increasing-sst/fv3gfs_run/sfc_dt_atmos.zarr")[["PRATEsfc", "TMPsfc"]]
        c48_real = xr.open_zarr(
            "gs://vcm-ml-raw-flexible-retention/2023-04-03-C48-CycleGAN-ramped-simulation/sfc_8xdaily.zarr"
        )[["PRATEsfc"]]
        c48_real = c48_real.chunk(
            {"time": 1488, "tile": 6, "grid_xt": 48, "grid_yt": 48}
        )
        try:
            c48_real.to_zarr(LOCAL_C48_FILENAME)
        except:
            shutil.rmtree(LOCAL_C48_FILENAME)
            raise
    c48_real = xr.open_zarr(LOCAL_C48_FILENAME)[["PRATEsfc"]]
    if not os.path.exists(LOCAL_C384_FILENAME):
        c384_real = convert_fine(
            xr.open_zarr(
                "gs://vcm-ml-raw-flexible-retention/2023-04-20-C384-CycleGAN-ramped-simulation/sfc_8xdaily_coarse.zarr"
            )
        )
        c384_real = c384_real.chunk(
            {"time": 744, "tile": 6, "grid_xt": 48, "grid_yt": 48}
        )
        try:
            c384_real.to_zarr(LOCAL_C384_FILENAME)
        except:
            shutil.rmtree(LOCAL_C384_FILENAME)
            raise
    c384_real = xr.open_zarr(LOCAL_C384_FILENAME)[["PRATEsfc"]]

    c48_real_all = c48_real.rename({"grid_xt": "x", "grid_yt": "y"})
    c384_real_all = c384_real.rename({"grid_xt": "x", "grid_yt": "y"})

    FILENAME = (
        f"./ramping_data/predicted-ramping-{BASE_NAME}-epoch_{EPOCH:03d}" + "-{res}.nc"
    )
    if not os.path.exists(FILENAME.format(res="c48")):
        c48_gen_all = cyclegan.predict(c384_real_all, reverse=True)
        c48_gen_all.to_netcdf(FILENAME.format(res="c48"))
    if not os.path.exists(FILENAME.format(res="c384")):
        c384_gen_all = cyclegan.predict(c48_real_all)
        c384_gen_all.to_netcdf(FILENAME.format(res="c384"))
    c48_gen_all = xr.open_dataset(FILENAME.format(res="c48"))
    c384_gen_all = xr.open_dataset(FILENAME.format(res="c384"))

    # plot_diurnal_land(c48, c384, c48_gen, c384_gen)

    # c48 = c48.coarsen(time=8).mean()
    # c384 = c384.coarsen(time=8).mean()
    # c48_gen = c48_gen.coarsen(time=8).mean()
    # c384_gen = c384_gen.coarsen(time=8).mean()

    c48_real = c48_real_all.isel(time=slice(1 * 365 * 8, 3 * 365 * 8))
    c384_real = c384_real_all.isel(time=slice(1 * 365 * 8, 3 * 365 * 8))
    c48_gen = c48_gen_all.isel(time=slice(1 * 365 * 8, 3 * 365 * 8))
    c384_gen = c384_gen_all.isel(time=slice(1 * 365 * 8, 3 * 365 * 8))

    animate_all(
        c48_real_all, c384_real_all, c48_gen_all, c384_gen_all, f"{BASE_NAME}-e{EPOCH}"
    )

    # time_slice = slice(0, 8)
    # animate_all(
    #     c48_real_all.isel(time=time_slice),
    #     c384_real_all.isel(time=time_slice),
    #     c48_gen_all.isel(time=time_slice),
    #     c384_gen_all.isel(time=time_slice),
    #     f"{BASE_NAME}-e{EPOCH}"
    # )

    # plot_weather_evaluation(c48_real_all.isel(time=slice(0, 14*8-1)), c384_real_all.isel(time=slice(0, 14*8-1)), c48_gen_all.isel(time=slice(0, 14*8-1)), c384_gen_all.isel(time=slice(0, 14*8-1)), f"{BASE_NAME}-e{EPOCH}")
    # plot_annual_means(c48_real_all, c384_real_all, c48_gen_all, c384_gen_all, f"{BASE_NAME}-e{EPOCH}")
    # plot_mean_all(c48_real, c384_real, c48_gen, c384_gen, f"{BASE_NAME}-e{EPOCH}")

    n_smoothing = 30
    plt.figure()
    plt.plot(
        c48_real.time.values,
        uniform_filter1d(
            c48_real[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C48",
    )
    plt.plot(
        c384_real.time.values,
        uniform_filter1d(
            c48_gen[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C48 gen",
    )
    plt.plot(
        c384_real.time.values,
        uniform_filter1d(
            c384_real[VARNAME].mean(dim=("tile", "x", "y")).values * TO_MM_DAY,
            size=n_smoothing,
        ),
        label="C384",
    )
    plt.plot(
        c48_real.time.values,
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
    plt.savefig(f"plots/ramping-{BASE_NAME}-e{EPOCH}.png", dpi=100)

    # n_bins = 100
    # bins = np.linspace(0, 1e-2, 101)
    # v1 = 1e-2 / n_bins ** 2
    # np.concatenate(
    #     [[-1.0, 0.0], np.logspace(np.log10(v1), np.log10(1e-2), n_bins - 1),]
    # )
    # time_bin = 30
    # c48_hist = np.zeros((len(c48.time) // time_bin, len(bins) - 1))
    # c48_gen_hist = np.zeros((len(c384.time) // (time_bin), len(bins) - 1))
    # c384_hist = np.zeros((len(c384.time) // (time_bin), len(bins) - 1))
    # c384_gen_hist = np.zeros((len(c48.time) // time_bin, len(bins) - 1))
    # for i_hist, i in enumerate(range(0, len(c48.time) - time_bin, time_bin)):
    #     c48_hist[i_hist, :] = np.histogram(
    #         c48[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
    #         bins=bins,
    #         density=True,
    #     )[0]
    #     c384_gen_hist[i_hist, :] = np.histogram(
    #         c384_gen[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
    #         bins=bins,
    #         density=True,
    #     )[0]
    # for i_hist, i in enumerate(range(0, len(c384.time) - time_bin, time_bin)):
    #     c384_hist[i_hist, :] = np.histogram(
    #         c384[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
    #         bins=bins,
    #         density=True,
    #     )[0]
    #     c48_gen_hist[i_hist, :] = np.histogram(
    #         c48_gen[VARNAME].isel(time=slice(i, i + time_bin)).values.flatten(),
    #         bins=bins,
    #         density=True,
    #     )[0]

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

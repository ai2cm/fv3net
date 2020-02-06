import fsspec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic
import xarray as xr

from vcm.calc import mass_integrate, r2_score
from vcm.calc.calc import solar_time
from vcm.convenience import round_time
from vcm.cubedsphere.constants import INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER, TILE_COORDS
from vcm.select import mask_to_surface_type
from vcm.visualize import plot_cube, mappable_var

kg_m2s_to_mm_day = (1e3 * 86400) / 997.
Lv = 2.5e6
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]

OUTPUT_FIG_DIR = "model_performance_plots"


def load_high_res_dataset(coarsened_hires_diags_path, init_times):
    fs = fsspec.filesystem("gs")
    ds_hires = xr.open_zarr(
        fs.get_mapper(coarsened_hires_diags_path), consolidated=True) \
        .rename({"time": INIT_TIME_DIM})
    ds_hires = ds_hires.assign_coords(
        {INIT_TIME_DIM: [round_time(t) for t in ds_hires[INIT_TIME_DIM].values],
         "tile": TILE_COORDS})
    ds_hires = ds_hires.sel({INIT_TIME_DIM: list(set(init_times))})
    if set(ds_hires[INIT_TIME_DIM].values) != set(init_times):
        raise ValueError(
            f"Timesteps {set(init_times)-set(ds_hires[INIT_TIME_DIM].values)}"
            f"are not matched in high res dataset.")
    ds_hires["P-E"] = SEC_PER_DAY * (
            ds_hires["PRATEsfc_coarse"] - ds_hires["LHTFLsfc_coarse"]/Lv)
    return ds_hires


def _merge_comparison_datasets(
        var, ds_pred, ds_data, ds_hires, grid):
    """ Makes a comparison dataset out of high res, C48 run, and prediction data
    for ease of plotting together.

    Args:
        ds_pred_unstacked: model prediction dataset
        ds_data_unstacked: target dataset
        ds_hires_unstacked: coarsened high res diagnostic data for comparison
        grid: dataset with lat/lon grid vars

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """
    slmsk = ds_data.isel({INIT_TIME_DIM: 0, COORD_Z_CENTER: -1}).slmsk
    src_dim_index = pd.Index(
        ["coarsened high res", "C48 run", "prediction"],
        name='dataset')
    ds_comparison = xr.merge(
        [xr.concat([ds_hires[var], ds_data[var], ds_pred[var]], src_dim_index), grid, slmsk])
    return ds_comparison


def make_r2_plot(
        ds_pred,
        ds_target,
        vars,
        output_dir,
        plot_filename="r2_vs_pressure_level.png",
        sample_dim=SAMPLE_DIM,
        save_fig=True,
        title=None
):
    matplotlib.rcParams['figure.dpi'] = 70
    if isinstance(vars, str):
        vars = [vars]
    x = ds_pred["pfull"].values
    for var in vars:
        y = r2_score(
            ds_target.stack(sample=STACK_DIMS)[var],
            ds_pred.stack(sample=STACK_DIMS)[var],
            sample_dim).values
        plt.plot(x, y, label=var)
    plt.legend()
    plt.xlabel("pressure level")
    plt.ylabel("$R^2$")
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def make_land_sea_r2_plot(
        ds_pred_sea, ds_pred_land, ds_target_sea, ds_target_land,
        vars,
        output_dir=OUTPUT_FIG_DIR,
        plot_filename="r2_vs_pressure_level_landsea.png",
        save_fig=True
):
    x = ds_pred_land.pfull.values
    colors=["blue", "orange"]
    for color, var in zip(colors, vars):

        y_sea = r2_score(
            ds_target_sea.stack(sample=STACK_DIMS)[var],
            ds_pred_sea.stack(sample=STACK_DIMS)[var],
            SAMPLE_DIM).values
        y_land = r2_score(
            ds_target_land.stack(sample=STACK_DIMS)[var],
            ds_pred_land.stack(sample=STACK_DIMS)[var],
            SAMPLE_DIM).values
        plt.plot(x, y_sea, color=color, alpha=0.7, label=f"{var}, sea", linestyle="--")
        plt.plot(x, y_land, color=color, alpha=0.7, label=f"{var}, land", linestyle=":")
    plt.legend()
    plt.xlabel("pressure level")
    plt.ylabel("$R^2$")
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def plot_diurnal_cycle(
        merged_ds,
        var,
        num_time_bins=24,
        title=None,
        output_dir=OUTPUT_FIG_DIR,
        plot_filename="r2_vs_pressure_level_landsea.png",
        save_fig=True
):
    matplotlib.rcParams['figure.dpi'] = 80

    for label in merged_ds["dataset"].values:
        ds = merged_ds.sel(dataset=label).stack(sample=STACK_DIMS).dropna("sample")
        solar_time = ds["solar_time"].values.flatten()
        data_var = ds[var].values.flatten()
        bin_means, bin_edges, _ = binned_statistic(
            solar_time, data_var, bins=num_time_bins)
        bin_centers = [
            0.5 *(bin_edges[i] + bin_edges[i+1]) for i in range(num_time_bins)]
        plt.plot(bin_centers, bin_means, label=label)
    plt.xlabel("solar time [hr]")
    plt.ylabel(var)
    plt.legend(loc="lower left")
    if title:
        plt.title(title)
    plt.show()




def make_all_plots(ds_pred, ds_target, ds_hires, grid):
    for ds in [ds_pred, ds_target, ds_hires]:
        if not set(STACK_DIMS).issubset(ds.dims):
            raise ValueError(f"Make sure all datasets are unstacked,"
                             "i.e. have original dimensions {STACK_DIMS}.")

    ds_pred["P-E"] = mass_integrate(-ds_pred["Q2"], ds_pred.delp) * kg_m2s_to_mm_day
    ds_target["P-E"] = mass_integrate(-ds_target["Q2"], ds_target.delp) * kg_m2s_to_mm_day

    slmsk = ds_target.isel({COORD_Z_CENTER: -1, INIT_TIME_DIM: 0}).slmsk

    ds_pred_sea = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "sea").drop("slmsk")
    ds_target_sea = mask_to_surface_type(xr.merge([ds_target, slmsk]), "sea").drop("slmsk")
    ds_pred_land = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "land").drop("slmsk")
    ds_target_land  = mask_to_surface_type(xr.merge([ds_target, slmsk]), "land").drop("slmsk")

    ds_pe = _merge_comparison_datasets(
        "P-E",
        ds_pred,
        ds_target,
        ds_hires,
        grid).drop("forecast_time")

    matplotlib.rcParams['figure.dpi'] = 200

    ds_pe["solar_time"] = solar_time(ds_pe)
    plot_diurnal_cycle(mask_to_surface_type(ds_pe, "sea"), "P-E", title="ocean")
    plot_diurnal_cycle(mask_to_surface_type(ds_pe, "land"), "P-E", title="land")

    fig_pe = plot_cube(mappable_var(ds_pe.isel({INIT_TIME_DIM: 0}), "P-E"), col="dataset")[0]
    fig_pe.savefig(os.path.join(OUTPUT_FIG_DIR, "P-E.png"))
    plt.show()

    make_r2_plot(
        ds_pred,
        ds_target,
        ["Q1", "Q2"],
        output_dir=OUTPUT_FIG_DIR,
        plot_filename="r2_vs_pressure_level_global.png",
        title="$R^2$, global")

    make_land_sea_r2_plot(
        ds_pred_sea, ds_pred_land, ds_target_sea, ds_target_land, vars=["Q1", "Q2"]
    )


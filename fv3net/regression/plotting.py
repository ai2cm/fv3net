import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic
import xarray as xr

from vcm.calc import mass_integrate, r2_score
from vcm.calc.calc import local_time
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_Z_CENTER,
    PRESSURE_GRID,
)
from vcm.cubedsphere.regridz import regrid_to_common_pressure
from vcm.select import mask_to_surface_type
from vcm.visualize import plot_cube, mappable_var

kg_m2s_to_mm_day = (1e3 * 86400) / 997.0
Lv = 2.5e6
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def _merge_comparison_datasets(var, ds_pred, ds_data, ds_hires, grid):
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
        ["coarsened high res", "C48 run", "prediction"], name="dataset"
    )
    ds_comparison = xr.merge(
        [
            xr.concat([ds_hires[var], ds_data[var], ds_pred[var]], src_dim_index),
            grid,
            slmsk,
        ]
    )
    return ds_comparison


def _make_r2_plot(
    ds_pred,
    ds_target,
    vars,
    output_dir,
    plot_filename="r2_vs_pressure_level.png",
    sample_dim=SAMPLE_DIM,
    save_fig=True,
    title=None,
):
    plt.clf()
    if isinstance(vars, str):
        vars = [vars]
    x = np.array(PRESSURE_GRID) / 100
    for var in vars:
        y = r2_score(
            regrid_to_common_pressure(ds_target[var], ds_target["delp"]).stack(
                sample=STACK_DIMS
            ),
            regrid_to_common_pressure(ds_pred[var], ds_pred["delp"]).stack(
                sample=STACK_DIMS
            ),
            sample_dim,
        ).values
        plt.plot(x, y, label=var)
    plt.legend()
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def _make_land_sea_r2_plot(
    ds_pred_sea,
    ds_pred_land,
    ds_target_sea,
    ds_target_land,
    vars,
    output_dir,
    plot_filename="r2_vs_pressure_level_landsea.png",
    save_fig=True,
):
    plt.clf()
    x = np.array(PRESSURE_GRID) / 100
    colors = ["blue", "orange"]
    for color, var in zip(colors, vars):
        y_sea = r2_score(
            regrid_to_common_pressure(ds_target_sea[var], ds_target_sea["delp"]).stack(
                sample=STACK_DIMS
            ),
            regrid_to_common_pressure(ds_pred_sea[var], ds_pred_sea["delp"]).stack(
                sample=STACK_DIMS
            ),
            SAMPLE_DIM,
        ).values
        y_land = r2_score(
            regrid_to_common_pressure(
                ds_target_land[var], ds_target_land["delp"]
            ).stack(sample=STACK_DIMS),
            regrid_to_common_pressure(ds_pred_land[var], ds_pred_land["delp"]).stack(
                sample=STACK_DIMS
            ),
            SAMPLE_DIM,
        ).values
        plt.plot(x, y_sea, color=color, alpha=0.7, label=f"{var}, sea", linestyle="--")
        plt.plot(x, y_land, color=color, alpha=0.7, label=f"{var}, land", linestyle=":")
    plt.legend()
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def _plot_diurnal_cycle(
    merged_ds,
    var,
    output_dir,
    num_time_bins=24,
    title=None,
    plot_filename="diurnal_cycle.png",
    save_fig=True,
):
    plt.clf()
    for label in merged_ds["dataset"].values:
        ds = merged_ds.sel(dataset=label).stack(sample=STACK_DIMS).dropna("sample")
        local_time = ds["local_time"].values.flatten()
        data_var = ds[var].values.flatten()
        bin_means, bin_edges, _ = binned_statistic(
            local_time, data_var, bins=num_time_bins
        )
        bin_centers = [
            0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(num_time_bins)
        ]
        plt.plot(bin_centers, bin_means, label=label)
    plt.xlabel("local_time [hr]")
    plt.ylabel(var)
    plt.legend(loc="lower left")
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def make_all_plots(ds_pred, ds_target, ds_hires, grid, output_dir):
    """ Makes figures for predictions on test data

    Args:
        ds_pred: unstacked dataset of prediction on test set
        ds_target: unstacked test data with target values
        ds_hires: unstacked coarsened high res diagnostic data
        grid: grid spec
        output_dir: location to write figures to

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """
    for ds in [ds_pred, ds_target, ds_hires]:
        if not set(STACK_DIMS).issubset(ds.dims):
            raise ValueError(
                f"Make sure all datasets are unstacked,"
                "i.e. have original dimensions {STACK_DIMS}."
            )
    report_sections = {}
    ds_pred["P-E"] = mass_integrate(-ds_pred["Q2"], ds_pred.delp) * kg_m2s_to_mm_day
    ds_target["P-E"] = (
        mass_integrate(-ds_target["Q2"], ds_target.delp) * kg_m2s_to_mm_day
    )

    # for convenience, separate the land/sea data
    slmsk = ds_target.isel({COORD_Z_CENTER: -1, INIT_TIME_DIM: 0}).slmsk
    ds_pred_sea = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "sea").drop("slmsk")
    ds_target_sea = mask_to_surface_type(xr.merge([ds_target, slmsk]), "sea").drop(
        "slmsk"
    )
    ds_pred_land = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "land").drop(
        "slmsk"
    )
    ds_target_land = mask_to_surface_type(xr.merge([ds_target, slmsk]), "land").drop(
        "slmsk"
    )
    ds_pe = _merge_comparison_datasets("P-E", ds_pred, ds_target, ds_hires, grid)

    # R^2 vs pressure plots
    matplotlib.rcParams["figure.dpi"] = 70
    _make_r2_plot(
        ds_pred,
        ds_target,
        ["Q1", "Q2"],
        output_dir=output_dir,
        plot_filename="r2_vs_pressure_level_global.png",
        title="$R^2$, global",
    )
    _make_land_sea_r2_plot(
        ds_pred_sea,
        ds_pred_land,
        ds_target_sea,
        ds_target_land,
        vars=["Q1", "Q2"],
        output_dir=output_dir,
        plot_filename="r2_vs_pressure_level_landsea.png",
    )
    report_sections["R^2 vs pressure levels"] = [
        "r2_vs_pressure_level_global.png",
        "r2_vs_pressure_level_landsea.png",
    ]

    # plot a variable across the diurnal cycle
    ds_pe["local_time"] = local_time(ds_pe)
    matplotlib.rcParams["figure.dpi"] = 80
    _plot_diurnal_cycle(
        mask_to_surface_type(ds_pe, "sea"),
        "P-E",
        title="ocean",
        output_dir=output_dir,
        plot_filename="diurnal_cycle_P-E_sea.png",
    )
    _plot_diurnal_cycle(
        mask_to_surface_type(ds_pe, "land"),
        "P-E",
        title="land",
        output_dir=output_dir,
        plot_filename="diurnal_cycle_P-E_land.png",
    )
    report_sections["Diurnal cycle"] = [
        "diurnal_cycle_P-E_sea.png",
        "diurnal_cycle_P-E_land.png",
    ]

    # map plot a variable and compare across prediction/ C48 /coarsened high res data
    matplotlib.rcParams["figure.dpi"] = 200
    plt.clf()
    time_label = ds_pe[INIT_TIME_DIM].values[0].strftime("%Y-%m-%d, %H:%M:%S")
    fig_pe = plot_cube(
        mappable_var(ds_pe.isel({INIT_TIME_DIM: 0}), "P-E"),
        col="dataset",
        cbar_label="P-E [mm/day]",
    )[0]
    plt.suptitle(time_label)
    fig_pe.savefig(os.path.join(output_dir, "P-E.png"))
    plt.show()
    report_sections["P-E"] = ["P-E.png"]

    return report_sections

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
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

from vcm.visualize.plot_diagnostics import plot_diurnal_cycle
from fv3net.diagnostics.data_funcs import merge_comparison_datasets


kg_m2s_to_mm_day = (1e3 * 86400) / 997.0
Lv = 2.5e6
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def _make_r2_plot(
    ds_pred,
    ds_target,
    vars,
    output_dir,
    plot_filename="r2_vs_pressure_level.png",
    sample_dim=SAMPLE_DIM,
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
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def plot_comparison_maps(
    ds_merged,
    var,
    output_dir,
    plot_filename,
    time_index_selection=None,
    plot_cube_kwargs=None,
):
    # map plot a variable and compare across prediction/ C48 /coarsened high res data
    matplotlib.rcParams["figure.dpi"] = 200
    plt.clf()
    plot_cube_kwargs = plot_cube_kwargs or {}

    if not time_index_selection:
        map_var = mappable_var(ds_merged.mean(INIT_TIME_DIM), var)
    else:
        map_var = mappable_var(
            ds_merged.isel({INIT_TIME_DIM: time_index_selection}), var
        )
        plot_cube_kwargs["row"] = INIT_TIME_DIM
    fig = plot_cube(map_var, col="dataset", **plot_cube_kwargs)[0]
    if isinstance(time_index_selection, int):
        time_label = (
            ds_merged[INIT_TIME_DIM]
            .values[time_index_selection]
            .strftime("%Y-%m-%d, %H:%M:%S")
        )
        plt.suptitle(time_label)
    fig.savefig(os.path.join(output_dir, plot_filename))
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
    ds_pe = merge_comparison_datasets(
        "P-E",
        [ds_pred, ds_target, ds_hires],
        ["prediction", "target C48", "coarsened high res"],
        grid,
        slmsk,
    )

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
    plot_diurnal_cycle(
        mask_to_surface_type(ds_pe, "sea"),
        "P-E",
        title="ocean",
        output_dir=output_dir,
        plot_filename="diurnal_cycle_P-E_sea.png",
    )
    plot_diurnal_cycle(
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
    plot_comparison_maps(
        ds_pe,
        "P-E",
        output_dir=output_dir,
        plot_filename="P-E_time_avg.png",
        time_index_selection=None,
        plot_cube_kwargs={"cbar_label": "time avg, P-E [mm/day]"},
    )
    plot_comparison_maps(
        ds_pe,
        "P-E",
        output_dir=output_dir,
        plot_filename="P-E_time_snapshots.png",
        time_index_selection=[0, 2],
        plot_cube_kwargs={"cbar_label": "timestep snapshot, P-E [mm/day]"},
    )
    report_sections["P-E"] = ["P-E_time_avg.png", "P-E_snapshots.png"]

    return report_sections

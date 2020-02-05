import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr

from vcm.calc import mass_integrate, r2_score
from vcm.cubedsphere.constants import INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER
from vcm.select import mask_to_surface_type

kg_m2s_to_mm_day = (1e3 * 86400) / 997.
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER]


def _merge_comparison_datasets(
        ds_pred, ds_data, ds_hires, grid):
    """

    Args:
        ds_pred_unstacked:
        ds_data_unstacked:
        ds_hires_unstacked:
        grid:

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """
    slmsk = ds_data.isel(initialization_time=0).slmsk
    src_dim_index = pd.Index(
        ["coarsened high res", "C48 run", "prediction"],
        name='dataset')
    ds_data = ds_data[["Q1", "Q2"]]
    ds_comparison = xr.merge(
        [xr.concat([ds_hires, ds_data, ds_pred], src_dim_index), grid, slmsk])
    return ds_comparison


def make_r2_plot(
        ds_pred,
        ds_target,
        vars,
        output_dir,
        plot_filename="r2_vs_pressure_level.png",
        sample_dim=SAMPLE_DIM,
):
    if isinstance(vars, str):
        vars = [vars]
    x = ds_pred["pfull"].values
    for var in vars:
        y = r2_score(
            ds_target.stack(sample=STACK_DIMS)[var],
            ds_pred[var].stack(sample=STACK_DIMS)[var],
            sample_dim).values
        plt.plot(x, y, label=var)
    plt.legend()
    plt.xlabel("pressure level")
    plt.ylabel("$R^2$")
    plt.savefig(os.path.join(output_dir, plot_filename))


def make_all_plots(ds_pred, ds_target, ds_hires, grid):
    for ds in [ds_pred, ds_target, ds_hires]:
        if set(ds.dims) != set(STACK_DIMS):
            raise ValueError(f"Make sure all datasets are unstacked,"
                             "i.e. have original dimensions {STACK_DIMS}.")
    slmsk = ds_pred.isel({COORD_Z_CENTER: -1, INIT_TIME_DIM: 0})

    ds_pred_sea = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "sea").drop("slmsk")
    ds_target_sea = mask_to_surface_type(xr.merge([ds_target, slmsk]), "sea").drop("slmsk")
    ds_pred_land = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "land ").drop("slmsk")
    ds_target_land  = mask_to_surface_type(xr.merge([ds_target, slmsk]), "land ").drop("slmsk")

    ds_merged = _merge_comparison_datasets(ds_pred, ds_target, ds_hires, grid)

    make_r2_plot(
        ds_pred,
        ds_target,
        ["Q1", "Q2"],
        output_dir="model_performance_plots",
        plot_filename="r2_vs_pressure_level_global.png")
    make_r2_plot(
        ds_pred_sea,
        ds_target_sea,
        ["Q1", "Q2"],
        output_dir="model_performance_plots",
        plot_filename="r2_vs_pressure_level_sea.png")
    make_r2_plot(
        ds_pred_land,
        ds_target_land,
        ["Q1", "Q2"],
        output_dir="model_performance_plots",
        plot_filename="r2_vs_pressure_level_land.png")


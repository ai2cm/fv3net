import fsspec
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr

from vcm.calc import mass_integrate, r2_score
from vcm.convenience import round_time
from vcm.cubedsphere.constants import INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER
from vcm.select import mask_to_surface_type
from vcm.visualize import plot_cube, mappable_var

kg_m2s_to_mm_day = (1e3 * 86400) / 997.
Lv = 2.5e6
SEC_PER_DAY = 86400

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER]


def _load_high_res_dataset(coarsened_hires_diags_path, init_times):
    fs = fsspec.filesystem("gs")
    ds_hires = xr.open_zarr(
        fs.get_mapper(coarsened_hires_diags_path), consolidated=True) \
        .rename({"time": INIT_TIME_DIM})
    ds_hires = ds_hires.assign_coords(
        {INIT_TIME_DIM: round_time(t) for t in ds_hires[INIT_TIME_DIM].values})
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
    src_dim_index = pd.Index(
        ["coarsened high res", "C48 run", "prediction"],
        name='dataset')
    ds_comparison = xr.merge(
        [xr.concat([ds_hires[var], ds_data[var], ds_pred[var]], src_dim_index), grid])
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

    ds_pred["P-E"] = mass_integrate(-ds_pred["Q2"], ds_pred.delp) * kg_m2s_to_mm_day
    ds_target["P-E"] = mass_integrate(-ds_target["Q2"], ds_target.delp) * kg_m2s_to_mm_day

    slmsk = ds_pred.isel({COORD_Z_CENTER: -1, INIT_TIME_DIM: 0})

    ds_pred_sea = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "sea").drop("slmsk")
    ds_target_sea = mask_to_surface_type(xr.merge([ds_target, slmsk]), "sea").drop("slmsk")
    ds_pred_land = mask_to_surface_type(xr.merge([ds_pred, slmsk]), "land ").drop("slmsk")
    ds_target_land  = mask_to_surface_type(xr.merge([ds_target, slmsk]), "land ").drop("slmsk")

    ds_pe = _merge_comparison_datasets(
        "P-E",
        ds_pred,
        ds_target,
        ds_hires,
        grid)
    plot_cube(mappable_var(ds_pe, "P-E").isel({INIT_TIME_DIM: 0}), column="dataset")

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


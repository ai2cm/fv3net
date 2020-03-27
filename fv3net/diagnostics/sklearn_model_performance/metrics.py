import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

import vcm
from vcm.calc import r2_score
from vcm.cubedsphere.regridz import regrid_to_common_pressure
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    PRESSURE_GRID,
    GRID_VARS,
)

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]
SAMPLE_DIM = "sample"

matplotlib.use("Agg")


def create_metrics_dataset(ds_pred, ds_fv3, ds_shield):

    ds_metrics = _r2_global_values(ds_pred, ds_fv3, ds_shield)
    for grid_var in GRID_VARS:
        ds_metrics[grid_var] = ds_pred[grid_var]

    for sfc_type in ["global", "sea", "land"]:
        for var in ["dQ1", "dQ2"]:
            ds_metrics[
                f"r2_{var}_pressure_levels_{sfc_type}"
            ] = _r2_pressure_level_metrics(
                vcm.mask_to_surface_type(ds_fv3, sfc_type)[var],
                vcm.mask_to_surface_type(ds_pred, sfc_type)[var],
                vcm.mask_to_surface_type(ds_fv3, sfc_type)["delp"],
            )
    ds_metrics = ds_metrics.assign_coords(
        {
            "target_dataset_names": [
                ds_target.dataset.values.item() for ds_target in [ds_fv3, ds_shield]
            ]
        }
    )

    for var in ["net_precipitation", "net_heating"]:
        for ds_target in [ds_fv3, ds_shield]:
            target_label = ds_target.dataset.values.item()
            ds_metrics[
                f"rmse_{var}_vs_{target_label}"
            ] = _root_mean_squared_error_metrics(ds_target[var], ds_pred[var])

    return ds_metrics


def plot_metrics(ds_metrics, output_dir, dpi_figures):
    report_sections = {}
    # R^2 vs pressure
    _plot_r2_pressure_profile(ds_metrics).savefig(
        os.path.join(output_dir, f"r2_pressure_levels.png"),
        dpi=dpi_figures["R2_pressure_profiles"],
    )
    report_sections["R^2 vs pressure levels"] = ["r2_vs_pressure_levels.png"]

    # RMSE maps
    report_sections["Root mean squared error maps"] = []
    for var in ["net_precipitation", "net_heating"]:
        for target_dataset_name in ds_metrics.target_dataset_names.values:
            filename = f"rmse_map_{var}_{target_dataset_name}.png"
            _plot_rmse_map(ds_metrics, var, target_dataset_name).savefig(
                os.path.join(output_dir, filename), dpi=dpi_figures["map_plot_single"]
            )
            report_sections["Root mean squared error maps"].append(filename)

    return report_sections


def _plot_rmse_map(ds, var, target_dataset_name):
    plt.close("all")
    data_var = f"rmse_{var}_vs_{target_dataset_name}"
    fig = vcm.plot_cube(
        vcm.mappable_var(ds[GRID_VARS + [data_var]], data_var), vmin=0, vmax=2
    )[0]
    return fig


def _plot_r2_pressure_profile(ds):
    plt.close("all")
    fig = plt.figure()
    for surface, surface_line in zip(["global", "land", "sea"], ["-", ":", "--"]):
        for var, var_color in zip(["dQ1", "dQ2"], ["orange", "blue"]):
            plt.plot(
                ds["pressure"],
                ds[f"r2_{var}_pressure_levels_{surface}"],
                color=var_color,
                linestyle=surface_line,
                label=f"{var}, {surface}",
            )
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    plt.legend()
    return fig


def _root_mean_squared_error_metrics(da_target, da_pred):
    rmse = np.sqrt((da_target - da_pred) ** 2).mean(INIT_TIME_DIM)
    return rmse


def _r2_pressure_level_metrics(da_target, da_pred, delp):
    pressure = np.array(PRESSURE_GRID) / 100
    target = regrid_to_common_pressure(da_target, delp).stack(sample=STACK_DIMS)
    prediction = regrid_to_common_pressure(da_pred, delp).stack(sample=STACK_DIMS)
    da = xr.DataArray(
        r2_score(target, prediction, "sample"),
        dims=["pressure"],
        coords={"pressure": pressure},
    )
    return da


def _r2_global_values(ds_pred, ds_fv3, ds_shield):
    """ Calculate global R^2 for net precipitation and heating against
    target FV3 dataset and coarsened high res dataset
    
    Args:
        ds ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    r2_summary = xr.Dataset()
    for var in ["net_heating", "net_precipitation"]:
        for sfc_type in ["global", "sea", "land"]:
            for ds_target in [ds_fv3, ds_shield]:
                target_label = ds_target.dataset.values.item()
                r2_summary[f"R2_{sfc_type}_{var}_vs_{target_label}"] = r2_score(
                    vcm.mask_to_surface_type(ds_target, sfc_type)[var].stack(
                        sample=STACK_DIMS
                    ),
                    vcm.mask_to_surface_type(ds_pred, sfc_type)[var].stack(
                        sample=STACK_DIMS
                    ),
                    "sample",
                ).values.item()
    return r2_summary

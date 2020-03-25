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
    COORD_Z_CENTER,
    VAR_LAT_CENTER,
    PRESSURE_GRID,
    GRID_VARS,
)
from . import (
    DATASET_NAME_PREDICTION, 
    DATASET_NAME_FV3_TARGET,
    DATASET_NAME_SHIELD_HIRES,
    DPI_FIGURES,
)

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]
SAMPLE_DIM = "sample"

matplotlib.use("Agg")


def create_metrics_dataset(ds):
    ds_metrics = _r2_global_values(ds)
    for grid_var in GRID_VARS:
        ds_metrics[grid_var] = ds[grid_var]
    ds_metrics["r2_dQ1_pressure_levels_global"] = _r2_pressure_level_metrics(ds, "dQ1")
    ds_metrics["r2_dQ2_pressure_levels_global"] = _r2_pressure_level_metrics(ds, "dQ2")
    ds_metrics["r2_dQ1_pressure_levels_sea"] = _r2_pressure_level_metrics(vcm.mask_to_surface_type(ds, "sea"), "dQ1")
    ds_metrics["r2_dQ2_pressure_levels_sea"] = _r2_pressure_level_metrics(vcm.mask_to_surface_type(ds, "sea"), "dQ2")
    ds_metrics["r2_dQ1_pressure_levels_land"] = _r2_pressure_level_metrics(vcm.mask_to_surface_type(ds, "land"), "dQ1")
    ds_metrics["r2_dQ2_pressure_levels_land"] = _r2_pressure_level_metrics(vcm.mask_to_surface_type(ds, "land"), "dQ2")
    ds_metrics["mse_net_precipitation_vs_fv3_target"] = _mean_squared_error_metrics(
        ds, "net_precipitation", target_dataset_name=DATASET_NAME_FV3_TARGET)
    ds_metrics["mse_net_precipitation_vs_shield"] = _mean_squared_error_metrics(
        ds, "net_precipitation", target_dataset_name=DATASET_NAME_SHIELD_HIRES)
    ds_metrics["mse_net_heating_vs_fv3_target"] = _mean_squared_error_metrics(
        ds, "net_heating", target_dataset_name=DATASET_NAME_FV3_TARGET)
    ds_metrics["mse_net_heating_vs_shield"] = _mean_squared_error_metrics(
        ds, "net_heating", target_dataset_name=DATASET_NAME_SHIELD_HIRES)
    return ds_metrics


def plot_metrics(ds, output_dir):
    report_sections = {}
    # R^2 vs pressure plots
    _plot_r2_global(ds).savefig(
        os.path.join(output_dir, f"r2_pressure_levels_global.png"),
        dpi=DPI_FIGURES["R2_pressure_profiles"])
    _plot_r2_land_sea(ds).savefig(
        os.path.join(output_dir, f"r2_pressure_levels_landsea.png"),
        dpi=DPI_FIGURES["R2_pressure_profiles"])
    report_sections["R^2 vs pressure levels"] = [
        "r2_vs_pressure_level_global.png",
        "r2_vs_pressure_level_landsea.png",
    ]

    # MSE maps
    report_sections["Mean squared error maps"] = []
    for var in ["net_precipitation", "net_heating"]:
        for target_dataset_name in ["fv3_target", "shield"]:
            filename = f"mse_map_{var}_{target_dataset_name}.png"
            _plot_mse_map(ds, var, target_dataset_name) \
                .savefig(
                    os.path.join(output_dir, filename), 
                    dpi=DPI_FIGURES["map_plot_single"])
            report_sections["Mean squared error maps"].append(filename)
    
    return report_sections
                    

def _plot_mse_map(ds, var, target_dataset_name):
    plt.close("all")
    data_var = f"mse_{var}_vs_{target_dataset_name}"
    fig = vcm.plot_cube(
        vcm.mappable_var(ds[GRID_VARS + [data_var]], data_var).mean(INIT_TIME_DIM),
        vmin=0,
        vmax=2,
    )[0]
    return fig


def _plot_r2_global(ds):
    plt.close("all")
    fig = plt.figure()
    for var in ["dQ1", "dQ2"]:
        plt.plot(ds["pressure"], ds[f"r2_{var}_pressure_levels_global"], label=var)
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    plt.legend()
    return fig


def _plot_r2_land_sea(ds):
    plt.close("all")
    fig = plt.figure()
    for surface, surface_line in zip(["land", "sea"], ["-", ":"]):
        for var, var_color in zip(["dQ1", "dQ2"], ["orange", "blue"]):
            plt.plot(
                ds["pressure"],
                ds[f"r2_{var}_pressure_levels_{surface}"], 
                color=var_color, 
                linestyle=surface_line, 
                label=f"{var}, {surface}")
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    plt.legend()
    return fig


def _mean_squared_error_metrics(
        ds, 
        var,
        target_dataset_name=DATASET_NAME_FV3_TARGET,
        prediction_dataset_name=DATASET_NAME_PREDICTION,
        sample_dim=SAMPLE_DIM,
):
    target = ds.sel(dataset=target_dataset_name)[var]
    prediction = ds.sel(dataset=prediction_dataset_name)[var]
    mse = np.sqrt((target - prediction)**2)
    return mse


def _r2_pressure_level_metrics(
        ds, 
        var,
        target_dataset_name=DATASET_NAME_FV3_TARGET,
        prediction_dataset_name=DATASET_NAME_PREDICTION,
        sample_dim=SAMPLE_DIM
):
    pressure = np.array(PRESSURE_GRID) / 100
    target = regrid_to_common_pressure(
        ds.sel(dataset=target_dataset_name)[var],
        ds.sel(dataset=target_dataset_name)["delp"]
    ).stack({sample_dim: STACK_DIMS})
    prediction = regrid_to_common_pressure(
        ds.sel(dataset=prediction_dataset_name)[var],
        ds.sel(dataset=prediction_dataset_name)["delp"]
    ).stack({sample_dim: STACK_DIMS})
    da = xr.DataArray(
        r2_score(target, prediction, sample_dim),
        dims=["pressure"],
        coords={"pressure": pressure}
    )
    return da


def _r2_global_values(ds):
    """ Calculate global R^2 for net precipitation and heating against
    target FV3 dataset and coarsened high res dataset
    
    Args:
        ds ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    r2_summary = xr.Dataset()
    for var in ["net_heating", "net_precipitation"]:
        r2_summary[f"R2_global_{var}_vs_target"] = r2_score(
            ds.sel(dataset=DATASET_NAME_FV3_TARGET)[var].stack(sample=STACK_DIMS),
            ds.sel(dataset=DATASET_NAME_PREDICTION)[var].stack(sample=STACK_DIMS),
            "sample",
        )
        r2_summary[f"R2_global_{var}_vs_hires"] = r2_score(
            ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)[var].stack(sample=STACK_DIMS),
            ds.sel(dataset=DATASET_NAME_PREDICTION)[var].stack(sample=STACK_DIMS),
            "sample",
        ).values.item()
        r2_summary[f"R2_sea_{var}_vs_target"] = r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_FV3_TARGET), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_sea_{var}_vs_hires"] = r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_SHIELD_HIRES), "sea")[
                var
            ].stack(sample=STACK_DIMS),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_land_{var}_vs_target"] = r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_FV3_TARGET), "land")[var].stack(
                sample=STACK_DIMS
            ),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "land")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_land_{var}_vs_hires"] = r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_SHIELD_HIRES), "land")[
                var
            ].stack(sample=STACK_DIMS),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "land")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
    return r2_summary

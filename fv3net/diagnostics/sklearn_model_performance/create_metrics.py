import numpy as np
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

METRICS_PRESSURE_LEVELS = [20000., 50000., 85000.]
SAMPLE_DIM = "sample"


def create_metrics_dataset(ds_pred, ds_fv3, ds_shield, names):

    ds_metrics = _r2_global_values(ds_pred, ds_fv3, ds_shield, names["stack_dims"])
    for grid_var in GRID_VARS:
        ds_metrics[grid_var] = ds_pred[grid_var]

    for sfc_type in ["global", "sea", "land"]:
        for var in ["dQ1", "dQ2"]:
            ds_metrics[
                f"r2_{var}_pressure_levels_{sfc_type}"
            ] = _r2_pressure_level_metrics(
                vcm.mask_to_surface_type(ds_fv3, sfc_type)[var],
                vcm.mask_to_surface_type(ds_pred, sfc_type)[var],
                vcm.mask_to_surface_type(ds_fv3, sfc_type)[names["var_pressure_thickness"]],
                names["stack_dims"]
            )
    # add a coordinate for target datasets so that the plot_metrics functions
    # can use it for labels
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

    for var in ["dQ1", "dQ2"]:
        var_rmse_pressure_levels = _rms_var_at_pressures(
            ds_target[var], 
            ds_pred[var], 
            ds_target[names["var_pressure_thickness"]], 
            METRICS_PRESSURE_LEVELS)
        for p in METRICS_PRESSURE_LEVELS:
            ds_metrics[f"rmse_{var}_{int(p/100)}hpa"] = var_rmse_pressure_levels.sel(pressure=p)

    return ds_metrics


def _root_mean_squared_error_metrics(da_target, da_pred):
    rmse = np.sqrt((da_target - da_pred) ** 2).mean(INIT_TIME_DIM)
    return rmse


def _r2_pressure_level_metrics(da_target, da_pred, delp, stack_dims):
    pressure = np.array(PRESSURE_GRID) / 100
    target = regrid_to_common_pressure(da_target, delp).stack(sample=stack_dims)
    prediction = regrid_to_common_pressure(da_pred, delp).stack(sample=stack_dims)
    da = xr.DataArray(
        r2_score(target, prediction, "sample"),
        dims=["pressure"],
        coords={"pressure": pressure},
    )
    return da


def _r2_global_values(ds_pred, ds_fv3, ds_shield, stack_dims):
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
                        sample=stack_dims
                    ),
                    vcm.mask_to_surface_type(ds_pred, sfc_type)[var].stack(
                        sample=stack_dims
                    ),
                    "sample",
                ).values.item()
    return r2_summary


def _rms_var_at_pressures(da_var_pred, da_var_target, delp, pressure_grid):
    """
    
    Args:
        da_var_pred (xr data array): predictied value
        da_var_target (xr data array): target value
        delp (xr data array): pressure level thickness
        pressure_grid (list(float)): pressures to interpolate to [pascals]
    
    Returns:
        xr data array: global mean RMSE of variable at each pressure
            in arg pressure_grid
    """
    target = regrid_to_common_pressure(da_var_target, delp, pressure_grid)
    pred = regrid_to_common_pressure(da_var_pred, delp, pressure_grid)
    rmse = _root_mean_squared_error_metrics(target, pred)
    return rmse.mean([dim for dim in rmse.dims if dim != "pressure"])


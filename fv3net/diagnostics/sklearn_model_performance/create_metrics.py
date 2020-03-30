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

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]
SAMPLE_DIM = "sample"


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

    for var in ["net_precipitation", "net_heating"]:
        for ds_target in [ds_fv3, ds_shield]:
            target_label = ds_target.dataset.values.item()
            ds_metrics[
                f"rmse_{var}_vs_{target_label}"
            ] = _root_mean_squared_error_metrics(ds_target[var], ds_pred[var])

    return ds_metrics


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

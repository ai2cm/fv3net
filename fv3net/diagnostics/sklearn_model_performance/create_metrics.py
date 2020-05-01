import numpy as np
from typing import Callable
import xarray as xr

import vcm
from vcm.calc import r2_score
from vcm.cubedsphere.regridz import regrid_to_common_pressure
from vcm.cubedsphere.constants import PRESSURE_GRID

SAMPLE_DIM = "sample"


def create_metrics_dataset(ds_pred, ds_fv3, ds_shield, names):

    ds_metrics = _r2_global_mean_values(ds_pred, ds_fv3, ds_shield, names["stack_dims"])
    for grid_var in names["grid_vars"]:
        ds_metrics[grid_var] = ds_pred[grid_var]

    for sfc_type in ["global", "sea", "land"]:
        for var in ["dQ1", "dQ2"]:
            ds_metrics[
                f"r2_{var}_pressure_levels_{sfc_type}"
            ] = _r2_pressure_level_metrics(
                vcm.mask_to_surface_type(ds_fv3, sfc_type)[var],
                vcm.mask_to_surface_type(ds_pred, sfc_type)[var],
                vcm.mask_to_surface_type(ds_fv3, sfc_type)[
                    names["var_pressure_thickness"]
                ],
                names["stack_dims"],
                names["coord_z_center"],
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
            ds_metrics[f"rmse_{var}_vs_{target_label}"] = _rmse(
                ds_target[var], ds_pred[var]
            ).mean(names["init_time_dim"])

    return ds_metrics


def calc_scalar_metrics(
    ds_pred, ds_fv3, ds_shield, init_time_dim, var_area, var_pressure_thickness
):
    rms_2d_vars = _metric_2d_global_mean(
        _rmse, ds_pred, ds_fv3, ds_shield, var_area, init_time_dim
    )
    bias_2d_vars = _metric_2d_global_mean(
        _bias, ds_pred, ds_fv3, ds_shield, var_area, init_time_dim
    )
    rms_3d_vars = _rmse_3d_col_weighted(
        ds_pred, ds_fv3, var_pressure_thickness, var_area, init_time_dim
    )
    return {**rms_2d_vars, **rms_3d_vars, **bias_2d_vars}


def _bias(da_target, da_pred):
    return np.mean(da_pred - da_target)


def _rmse(da_target, da_pred):
    rmse = np.sqrt((da_target - da_pred) ** 2)
    return rmse


def _metric_2d_global_mean(
    metric_func: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    ds_pred: xr.Dataset,
    ds_fv3: xr.Dataset,
    ds_shield: xr.Dataset,
    var_area: str = "area",
    init_time_dim: str = "initial_time",
):
    metric_name = metric_func.__name__.strip("_")
    metrics = {}
    area_weights = ds_pred[var_area] / (ds_pred[var_area].mean())
    pred_labels = ["ML+physics", "physics", "globalmean"]
    for var, units in zip(["net_precipitation", "net_heating"], ["mm/d", "W/m^2"]):
        # this loop is over target dataset so that mean can be taken for one of the comparisons
        for da_target, target_label in zip(
            [ds_fv3[var], ds_shield[var]], ["target", "hires"]
        ):
            for da_pred, pred_label in zip(
                [
                    ds_pred[var],
                    ds_fv3[f"{var}_physics"],
                    da_target.mean().values.item(),
                ],
                pred_labels,
            ):
                # compare to ML predicion, model physics only prediction, variable average
                global_metric = (
                    (metric_func(da_target, da_pred) * area_weights)
                    .mean()
                    .values.item()
                )
                metrics[f"{metric_name}/{var}/{pred_label}_vs_{target_label}"] = {
                    metric_name: global_metric,
                    "units": units,
                }
    return metrics


def _rmse_mass_avg(
    da_target,
    da_pred,
    delp,
    area,
    coord_z_center="z",
    coords_horizontal=["x", "y", "tile"],
):
    mse = (da_target - da_pred) ** 2
    num2 = ((mse * delp).sum([coord_z_center]) * area).sum(coords_horizontal)
    denom2 = (delp.sum([coord_z_center]) * area).sum(coords_horizontal)
    return np.sqrt(num2 / denom2)


def _rmse_3d_col_weighted(
    ds_pred,
    ds_fv3,
    var_pressure_thickness="pressure_thickness_of_atmospheric_layer",
    var_area="area",
    init_time_dim="initial_time",
):
    rmse_metrics = {}
    delp, area = ds_pred[var_pressure_thickness], ds_pred[var_area]
    pred_labels = ["ML+physics", "physics", "globalmean"]
    for total_var, phys_var, units in zip(
        ["Q1", "Q2"], ["pQ1", "pQ2"], ["K/s", "kg/kg/s"]
    ):
        da_target = ds_fv3[total_var]
        for da_pred, pred_label in zip(
            [ds_pred[total_var], ds_fv3[phys_var], ds_fv3[total_var].mean()],
            pred_labels,
        ):
            rmse_weighted = (
                _rmse_mass_avg(da_target, da_pred, delp, area)
                .mean(init_time_dim)
                .values.item()
            )
            rmse_metrics[f"rms_col_int/{total_var}/{pred_label}_vs_target"] = {
                "rms": rmse_weighted,
                "units": units,
            }
    return rmse_metrics


def _r2_pressure_level_metrics(da_target, da_pred, delp, stack_dims, coord_z_center):
    pressure = np.array(PRESSURE_GRID) / 100
    target = regrid_to_common_pressure(da_target, delp, coord_z_center).stack(
        sample=stack_dims
    )
    prediction = regrid_to_common_pressure(da_pred, delp, coord_z_center).stack(
        sample=stack_dims
    )
    da = xr.DataArray(
        r2_score(target, prediction, "sample"),
        dims=["pressure"],
        coords={"pressure": pressure},
    )
    return da


def _r2_global_mean_values(ds_pred, ds_fv3, ds_shield, stack_dims):
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
                r2 = r2_score(
                    vcm.mask_to_surface_type(ds_target, sfc_type)[var].stack(
                        sample=stack_dims
                    ),
                    vcm.mask_to_surface_type(ds_pred, sfc_type)[var].stack(
                        sample=stack_dims
                    ),
                    "sample",
                )
                r2_summary[
                    f"r2_{var}_{sfc_type}_ML_vs_{target_label}"
                ] = r2.values.item()
    return r2_summary

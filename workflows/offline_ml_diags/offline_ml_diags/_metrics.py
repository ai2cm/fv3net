import logging
import numpy as np
from typing import Sequence, Callable, Union, Mapping
import xarray as xr

from vcm import safe
from vcm.cubedsphere import regrid_to_common_pressure


logging.getLogger(__name__)

# Variables predicted by model
ML_VARS = ["dQ1", "dQ2"]
# Variables to calculate RMSE and bias of
PRESSURE_LEVEL_METRIC_VARS = ["dQ1", "dQ2", "Q1", "Q2"]
COLUMN_INTEGRATED_METRIC_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]

# Dimension/ coord/ variable parameter defaults
PREDICT_COORD = "predict"
TARGET_COORD = "target"
DERIVATION_DIM = "derivation"
PRESSURE_DIM = "pressure"
VERTICAL_DIM = "z"
AREA_VAR = "area"
DELP_VAR = "pressure_thickness_of_atmospheric_layer"
TOA_PRESSURE = 300.0  # Pa
VERTICAL_PROFILE_MEAN_DIMS = ("time", "x", "y", "tile")


def calc_metrics(
    ds: xr.Dataset,
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
    pressure_dim: str = PRESSURE_DIM,
    vertical_dim: str = VERTICAL_DIM,
    area_var: str = AREA_VAR,
    delp_var: str = DELP_VAR,
    toa_pressure: float = TOA_PRESSURE,
    vertical_profile_mean_dims: Sequence[str] = VERTICAL_PROFILE_MEAN_DIMS,
) -> xr.Dataset:
    """Routine for computing ML prediction metrics (_bias, _rmse]) on a dataset of
    variables, assumed to include variables in
    {SCALAR_METRIC_VARS, VERTICAL_METRIC_VARS} as well as area and delp
    """
    derivation_kwargs = {
        "predict_coord": predict_coord,
        "target_coord": target_coord,
        "derivation_dim": derivation_dim,
    }

    ds = _insert_weights(ds, vertical_dim, area_var, delp_var)
    area_weights = ds[area_var] / (ds[area_var].mean())
    delp_weights = ds[delp_var] / ds[delp_var].mean(vertical_dim)

    # scalar quantity metrics calculated for 2d column integrated vars
    scalar_metrics_column_integrated_vars = _calc_same_dims_metrics(
        ds,
        dim_tag="scalar",
        vars=COLUMN_INTEGRATED_METRIC_VARS,
        weights=[area_weights],
        mean_dim_vars=None,
        **derivation_kwargs,
    )

    # scalar quantity metrics are calculated at vertical levels first,
    # then weighted by pressure level and then integrated
    scalar_column_integrated_metrics = _calc_same_dims_metrics(
        ds,
        dim_tag="scalar",
        vars=PRESSURE_LEVEL_METRIC_VARS,
        weights=[area_weights, delp_weights],
        mean_dim_vars=None,
        **derivation_kwargs,
    )

    ds_regrid_z = _regrid_dataset_zdim(
        ds, vertical_dim, pressure_dim, delp_var, toa_pressure, **derivation_kwargs
    )

    pressure_level_metrics = _calc_same_dims_metrics(
        ds_regrid_z,
        dim_tag="pressure_level",
        vars=PRESSURE_LEVEL_METRIC_VARS,
        weights=[area_weights],
        mean_dim_vars=vertical_profile_mean_dims,
        **derivation_kwargs,
    )
    ds = xr.merge(
        [
            scalar_metrics_column_integrated_vars,
            scalar_column_integrated_metrics,
            pressure_level_metrics,
        ]
    )
    _insert_r2(ds)
    return ds


def _regrid_dataset_zdim(
    ds: xr.Dataset,
    vertical_dim: str = VERTICAL_DIM,
    pressure_dim: str = PRESSURE_DIM,
    delp_var: str = DELP_VAR,
    toa_pressure: float = TOA_PRESSURE,
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
) -> xr.Dataset:
    # have to separate the derivation coordinates before interpolating
    # to regridded pressure
    regridded_datasets = []
    vertical_dim_vars = [var for var in ds.data_vars if vertical_dim in ds[var].dims]
    ds_2d = ds.drop(vertical_dim_vars)
    ds_3d = safe.get_variables(ds, vertical_dim_vars)

    for derivation_coord in [target_coord, predict_coord]:
        ds_regrid = ds_3d.sel({derivation_dim: derivation_coord})
        for var in vertical_dim_vars:
            ds_regrid[var] = regrid_to_common_pressure(
                field=ds_regrid[var],
                delp=ds[delp_var],
                coord_z_center=vertical_dim,
                new_vertical_dim=pressure_dim,
            )
        regridded_datasets.append(ds_regrid)
    return xr.merge([ds_2d, xr.concat(regridded_datasets, dim=derivation_dim)])


def _calc_same_dims_metrics(
    ds: xr.Dataset,
    dim_tag: str,
    vars: Sequence[str],
    weights: Sequence[xr.DataArray] = None,
    mean_dim_vars: Sequence[str] = None,
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
) -> xr.Dataset:
    """Computes a set of metrics that all have the same dimension,
    ex. mean vertical error profile on pressure levels, or global mean scalar error
    Args:
        ds: test dataset
        dim_tag: describe the dimensions of the metrics returned. e.g. "scalar", "z"
        vars: data variables to calculate metrics on
        weight_vars: data variables to use as weights
        mean_dim_vars: Sequence of dimensions to take each weighted mean over
            for final value. Must match order of weights.
            Defaults to None (mean over all dims).
        target_coord: derivation coord for TARGET_COORD values in metric comparison.
        predict_coord: derivation coord for "prediction".
        derivation_dim: dimension name for derivation (comparision)

    Returns:
        dataset of metrics
    """
    weights = weights or [1.0]
    metric_comparison_coords = [(target_coord, predict_coord), (target_coord, "mean")]
    ds = _insert_means(
        ds, vars, predict_coord, target_coord, derivation_dim, weights, mean_dim_vars
    )
    metrics = xr.Dataset()
    for var in vars:
        for metric_func in (_bias, _rmse):
            for comparison in metric_comparison_coords:
                metric = _calc_metric(
                    ds,
                    metric_func,
                    var,
                    derivation_dim,
                    *comparison,
                    weights,
                    mean_dim_vars,
                )
                metrics[f"{dim_tag}/{metric.name}"] = metric
    return metrics


def _insert_r2(
    ds: xr.Dataset,
    rmse_coord: str = "rmse",
    r2_coord: str = "r2",
    predict_coord: str = "predict",
    target_coord: str = "target",
):
    rmse_vars = [
        var
        for var in ds.data_vars
        if (var.endswith(f"{predict_coord}_vs_{target_coord}") and rmse_coord in var)
    ]
    for rmse_var in rmse_vars:
        name_pieces = rmse_var.split("/")
        std_var = "/".join(name_pieces[:-1] + [f"mean_vs_{target_coord}"])
        r2_var = "/".join([s if s != rmse_coord else r2_coord for s in name_pieces])
        ds[r2_var] = 1.0 - (ds[rmse_var] / ds[std_var]) ** 2


def _insert_weights(
    ds,
    vertical_dim: str = VERTICAL_DIM,
    area_var: str = AREA_VAR,
    delp_var: str = DELP_VAR,
):
    ds[f"{area_var}_weights"] = ds[area_var] / (ds[area_var].mean())
    ds[f"{delp_var}_weights"] = ds[delp_var] / ds[delp_var].mean(vertical_dim)
    return ds


def _insert_means(
    ds: xr.Dataset,
    var_names: Sequence[str],
    predict_coord: str = PREDICT_COORD,
    target_coord: str = TARGET_COORD,
    derivation_dim: str = DERIVATION_DIM,
    weights: Sequence[xr.DataArray] = None,
    mean_dims: Sequence[str] = None,
) -> xr.Dataset:
    weights = weights or [1.0]
    for var in var_names:
        da = ds[var].sel({derivation_dim: [target_coord, predict_coord]})
        da_mean = _weighted_average(
            da.sel({derivation_dim: target_coord}), weights, mean_dims
        ).assign_coords({derivation_dim: "mean"})
        da = xr.concat([da, da_mean], dim=derivation_dim)
        ds = ds.drop([var])
        ds = ds.merge(da)
    return ds


def _calc_metric(
    ds: xr.Dataset,
    metric_func: Callable[
        [xr.DataArray, xr.DataArray, xr.DataArray, Sequence[str]], xr.DataArray
    ],
    var: str,
    derivation_dim: str,
    target_coord: str,
    predict_coord: str,
    weights: Sequence[xr.DataArray] = None,
    mean_dims: Sequence[str] = None,
) -> xr.DataArray:
    """helper function to calculate arbitrary metrics given the variable, target, and
    prediction coords

    Args:
        ds: Dataset with variables to calculate metrics on. Variables
            should have a DERIVATION dim with coords denoting the sets for comparison,
            e.g. "prediction", TARGET_COORD
        metric_func: metric calculation function.
            One of {_rmse, _biase}.
        var: Variable name to calculate metric for
        target_coord: derivation coord for TARGET_COORD values in metric comparison.
        predict_coord: derivation coord for "prediction".
        derivation_dim: dimension name for derivation (comparision)
        weights: data arrays to weight the metric over before taking mean.
        mean_dims: Sequence of dimensions to take each weighted mean over
            for final value. Must match order of weights. Defaults to None (all)

    Returns:
        xr.DataArray: data array of metric values
    """
    da_target = ds[var].sel({derivation_dim: target_coord})
    da_predict = ds[var].sel({derivation_dim: predict_coord})
    metric = metric_func(da_target, da_predict)
    metric_weighted_average = _weighted_average(metric, weights, mean_dims)
    metric_name = (
        f"{metric_func.__name__.strip('_')}/{var}/{predict_coord}_vs_{target_coord}"
    )
    return metric_weighted_average.rename(metric_name)


def _bias(da_target: xr.DataArray, da_pred: xr.DataArray,) -> xr.DataArray:
    return da_pred - da_target


def _rmse(
    da_target: xr.DataArray, da_pred: xr.DataArray,
):
    return np.sqrt((da_target - da_pred) ** 2)


def _weighted_average(
    data: Union[xr.DataArray, xr.Dataset],
    weights: Sequence[xr.DataArray],
    mean_dims: Sequence[str] = None,
):
    # Differs from diagnostics_utils.weighted_average in that
    # this does multiple weightings (e.g. area + delp) in one go
    # NOTE: this assumes the variables used in weighting have already
    # been transformed into weights!
    for weight in weights:
        data *= weight
    return data.mean(dim=mean_dims, skipna=True)


def _get_r2_string(
    metrics: Mapping[str, Mapping[str, float]],
    var: str,
    predict_coord: str = "predict",
    target_coord: str = "target",
    precision=2,
):
    value = metrics[f"scalar/r2/{var}/{predict_coord}_vs_{target_coord}"]["mean"]
    std = metrics[f"scalar/r2/{var}/{predict_coord}_vs_{target_coord}"]["std"]
    return f"{value:.{precision}f} +/- {std:.{precision}f}"


def _get_bias_string(
    metrics: Mapping[str, Mapping[str, float]],
    var: str,
    predict_coord: str = "predict",
    target_coord: str = "target",
    precision=2,
):
    value = metrics[f"scalar/bias/{var}/{predict_coord}_vs_{target_coord}"]["mean"]
    std = metrics[f"scalar/bias/{var}/{predict_coord}_vs_{target_coord}"]["std"]
    return f"{value:.{precision}f} +/- {std:.{precision}f}"

import logging
import numpy as np
from typing import Mapping, Sequence, Callable
import xarray as xr

from fv3net.regression.sklearn import TARGET_COORD, PREDICT_COORD, DERIVATION_DIM
from vcm.cubedsphere import regrid_to_common_pressure
from vcm import safe

logging.getLogger(__name__)

# Variables predicted by model
ML_VARS = ["dQ1", "dQ2"]
# Variables to calculate RMSE and bias of

PRESSURE_LEVEL_METRIC_VARS = ["dQ1", "dQ2", "Q1", "Q2"]
SCALAR_METRIC_VARS = PRESSURE_LEVEL_METRIC_VARS + [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]
# Comparison pairs for RMSE and bias. Truth/target first.
METRIC_COMPARISON_COORDS = [(TARGET_COORD, PREDICT_COORD), (TARGET_COORD, "mean")]
VERTICAL_PROFILE_MEAN_DIMS = ["time", "x", "y", "tile"]
DELP_VAR = "pressure_thickness_of_atmospheric_layer"
AREA_VAR = "area"
DELP_WEIGHT_VAR = f"{DELP_VAR}_weights"
AREA_WEIGHT_VAR = f"{AREA_VAR}_weights"
VERTICAL_DIM = "z"


def calc_metrics(ds: xr.Dataset) -> xr.Dataset:
    """Routine for computing ML prediction metrics (_bias, _rmse]) on a dataset of
    variables, assumed to include variables in
    {SCALAR_METRIC_VARS, VERTICAL_METRIC_VARS} as well as area and delp
    """

    ds = _insert_weights(ds).chunk({'z': -1})
    scalar_metrics = _calc_same_dims_metrics(
        ds,
        dim_tag="scalar",
        vars=SCALAR_METRIC_VARS,
        weights=[AREA_WEIGHT_VAR],
        mean_dim_vars=None)

    ds_regrid_z = _regrid_dataset_to_pressure_levels(
        ds,
        PRESSURE_LEVEL_METRIC_VARS,
        ds[DELP_VAR]
    )

    pressure_level_metrics = _calc_same_dims_metrics(
        ds_regrid_z,
        dim_tag="z",
        vars=PRESSURE_LEVEL_METRIC_VARS,
        weights=[AREA_WEIGHT_VAR, DELP_WEIGHT_VAR],
        mean_dim_vars=["time", "x", "y", "tile"])
    return xr.merge([scalar_metrics, pressure_level_metrics])


def _regrid_dataset_to_pressure_levels(
        ds: xr.Dataset, regrid_vars: Sequence[str], da_delp: xr.DataArray):
    for var in regrid_vars:
        ds[var] = regrid_to_common_pressure(
            ds[var],
            da_delp,
            VERTICAL_DIM,
        )
    return ds


def _calc_same_dims_metrics(
        ds: xr.Dataset,
        dim_tag: str,
        vars: Sequence[str],
        weights: Sequence[str],
        mean_dim_vars: Sequence[str] = None) -> xr.Dataset:
    """Computes a set of metrics that all have the same dimension,
    ex. mean vertical error profile on pressure levels, or global mean scalar error
    Args:
        ds: test dataset
        dim_tag: describe the dimensions of the metrics returned. e.g. "scalar", "z"
        vars (Sequence[str]): data variables to calculate metrics on
        weights (Sequence[str]): data variables to use as weights
        mean_dim_vars (Sequence[str], optional): dimensions to take mean over
            for final value. Defaults to None (mean over all dims).

    Returns:
        dataset of metrics
    """
    metric_kwargs = {"weights": [ds[weight] for weight in weights]}
    ds = _insert_means(ds, vars, **metric_kwargs)
    metrics = xr.Dataset()
    for var in vars:
        for metric_func in (_bias, _rmse):
            for comparison in METRIC_COMPARISON_COORDS:
                metric = _calc_metric(ds, metric_func, var, *comparison, metric_kwargs)
                metrics[f"{dim_tag}/{metric.name}"] = metric
    return metrics


def _insert_weights(ds):
    ds[f"{AREA_VAR}_weights"] = ds[AREA_VAR] / (ds[AREA_VAR].mean())
    ds[f"{DELP_VAR}_weights"] = ds[DELP_VAR] / ds[DELP_VAR].mean(VERTICAL_DIM)
    return ds


def _insert_means(
    ds: xr.Dataset, var_names: Sequence[str], weights: Sequence[xr.DataArray] = None, mean_dims: Sequence[str] = None,
) -> xr.Dataset:
    for var in var_names:
        da = ds[var].sel({DERIVATION_DIM: [TARGET_COORD, PREDICT_COORD]})
        weights = [1.0] if weights is None else weights
        
        da_mean = da.sel({DERIVATION_DIM: TARGET_COORD})
        for weight in weights:
            da_mean *= weight
        da_mean = da_mean \
            .mean(dim=mean_dims) \
            .assign_coords({DERIVATION_DIM: "mean"})
        da = xr.concat([da, da_mean], dim=DERIVATION_DIM)
        ds = ds.drop([var])
        ds = ds.merge(da)
    return ds


def _calc_metric(
    ds: xr.Dataset,
    metric_func: Callable[
        [xr.DataArray, xr.DataArray, xr.DataArray, Sequence[str]], xr.DataArray
    ],
    var: str,
    target_coord: str,
    predict_coord: str,
    metric_kwargs: Mapping = None,
) -> xr.DataArray:
    """helper function to calculate arbitrary metrics given the variable, target, and
    prediction coords

    Args:
        ds (xr.Dataset): Dataset with variables to calculate metrics on. Variables
            should have a DERIVATION dim with coords denoting the sets for comparison,
            e.g. "prediction", "target"
        metric_func (Callable, xr.DataArray]): metric calculation function.
            One of {_rmse, _biase}.
        var (str): Variable name to calculate metric for
        target_coord (str): DERIVATION coord for "target" values in metric comparison.
        predict_coord (str): DERIVATION coord for "prediction".
        metric_kwargs (Mapping, optional): [description]. Defaults to None.

    Returns:
        xr.DataArray: data array of metric values
    """
    da_target = ds[var].sel({DERIVATION_DIM: target_coord})
    da_predict = ds[var].sel({DERIVATION_DIM: predict_coord})
    metric = metric_func(da_target, da_predict, **(metric_kwargs or {}))
    metric_name = (
        f"{metric_func.__name__.strip('_')}/{var}/{predict_coord}_vs_{target_coord}"
    )
    return metric.rename(metric_name)


def _bias(
    da_target: xr.DataArray,
    da_pred: xr.DataArray,
    weights: Sequence[xr.DataArray] = None,
    mean_dims: Sequence[str] = None,
) -> xr.DataArray:
    bias = da_pred - da_target
    if weights is not None:
        for weight_da in weights:
            bias *= weight_da
    return bias.mean(dim=mean_dims)


def _rmse(
    da_target: xr.DataArray,
    da_pred: xr.DataArray,
    weights: Sequence[xr.DataArray] = None,
    mean_dims: Sequence[str] = None,
):
    se = (da_target - da_pred) ** 2
    if weights is not None:
        for weight_da in weights:
            se *= weight_da
    return np.sqrt(se.mean(dim=mean_dims))

import numpy as np
from typing import Mapping, Sequence, Callable
import logging
from fv3net.regression.sklearn import TARGET_COORD, PREDICT_COORD, DERIVATION_DIM
from diagnostics_utils import insert_column_integrated_vars
import xarray as xr

logging.getLogger(__name__)

# Variables predicted by model
ML_VARS = ["dQ1", "dQ2"]
# Variables to calculate RMSE and bias of
METRIC_VARS = ["dQ1", "dQ2", "column_integrated_dQ1", "column_integrated_dQ2"]
# Comparison pairs for RMSE and bias. Truth/target first.
METRIC_COMPARISON_COORDS = [(TARGET_COORD, PREDICT_COORD), (TARGET_COORD, "mean")]
VERTICAL_PROFILE_MEAN_DIMS = ["time", "x", "y", "tile"]


def calc_metrics(ds: xr.Dataset) -> xr.Dataset:
    metrics = xr.Dataset()
    metric_kwargs = {"weights": area_weights}
    for var in METRIC_VARS:
        for metric_func in [_bias, _rmse]:
            for comparison in METRIC_COMPARISON_COORDS:
                metric = _calc_metric(ds, metric_func, var, *comparison, metric_kwargs)
                metrics[metric.name] = metric
    return metrics


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
    weights: xr.DataArray = None,
    mean_dims: Sequence[str] = None,
) -> xr.DataArray:
    bias = da_pred - da_target
    if weights is not None:
        bias *= weights
    return bias.mean(dim=mean_dims)


def _rmse(
    da_target: xr.DataArray,
    da_pred: xr.DataArray,
    weights: xr.DataArray = None,
    mean_dims: Sequence[str] = None,
):
    se = (da_target - da_pred) ** 2
    if weights is not None:
        se *= weights
    return np.sqrt(se.mean(dim=mean_dims))

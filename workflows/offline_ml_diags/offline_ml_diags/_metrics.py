import numpy as np
from typing import Mapping, Sequence, Callable
import logging
from fv3net.regression.sklearn import TARGET_COORD, PREDICT_COORD, DERIVATION_DIM
import xarray as xr
from ._metrics_config import SCALAR_METRIC_KWARGS
from ._utils import insert_additional_variables


logging.getLogger(__name__)
ConfigMapping = Mapping[str, Mapping[str, Mapping[str, Sequence]]]

# Variables predicted by model
ML_VARS = ["dQ1", "dQ2", "Q1", "Q2"]
# Variables to calculate RMSE and bias of
METRIC_VARS = [
    "Q1",
    "Q2",
    "dQ1",
    "dQ2",
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]

# Comparison pairs for RMSE and bias. Truth/target first.
METRIC_COMPARISON_COORDS = [(TARGET_COORD, PREDICT_COORD), (TARGET_COORD, "mean")]
VERTICAL_PROFILE_MEAN_DIMS = ["time", "x", "y", "tile"]


def calc_metrics(
    dataset_sequence: Sequence[xr.Dataset], area: xr.DataArray
) -> Mapping[str, Mapping[str, float]]:
    """Calculate metrics over a sequence of batches and return the
    mean/std over all batches.

    Args:
        dataset_sequence (Sequence[xr.Dataset]): Sequence of batched data to be
        iterated over to calculate batch metrics and their mean/std
        area: dataarray for grid area variable

    Returns:
        Mapping[str, Mapping[str, float]]: Dict of metrics and their mean/std
        over batches
    """
    metrics_batch_collection = []
    
    for i, ds_batch in enumerate(dataset_sequence):
        # batch metrics are kept in dataset format for ease of concatting
        ds = insert_additional_variables(ds_batch, area)
        batch_metrics = _calc_batch_metrics(ds, SCALAR_METRIC_KWARGS).assign_coords({"batch": i})
        metrics_batch_collection.append(batch_metrics)
    ds = xr.concat(metrics_batch_collection, dim="batch")
    metrics = {
        var: {"mean": np.mean(ds[var].values), "std": np.std(ds[var].values)}
        for var in ds.data_vars
    }
    return metrics


def _calc_batch_metrics(ds: xr.Dataset, config: ConfigMapping) -> xr.Dataset:
    metrics = xr.Dataset()

    for var, metric_config in config.items():
        for metric_name, kwargs in metric_config.items():
            metric_func = globals()[f"_{metric_name}"]
            for comparison in METRIC_COMPARISON_COORDS:
                metric = _calc_quantity_metric(ds, metric_func, var, *comparison, kwargs)
                metrics[metric.name] = metric

    return metrics


def _calc_quantity_metric(
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
    # fill weights kwarg with data arrays, if present
    metric_kwargs_copy = dict(metric_kwargs) or {}
    if "weights_variables" in metric_kwargs:
        metric_kwargs_copy["weights"] = [
            ds[weight_var] for weight_var in metric_kwargs["weights_variables"]]
        del metric_kwargs_copy["weights_variables"]
    metric = metric_func(da_target, da_predict, **metric_kwargs_copy)

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

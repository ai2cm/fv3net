import numpy as np
from typing import Mapping, Sequence, Callable
import logging
from fv3net.regression.sklearn import TARGET_COORD, PREDICT_COORD, DERIVATION_DIM
from diagnostics_utils import insert_column_integrated_vars
import xarray as xr
from ._metrics_config import SCALAR_METRIC_KWARGS, fill_config_weights

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
AREA_VAR = "area"
DELP_VAR = "pressure_thickness_of_atmospheric_layer"


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
        ds = xr.merge([area, ds_batch]).load()
        ds = _insert_additional_variables(ds)
        config = fill_config_weights(ds, SCALAR_METRIC_KWARGS)
        print(SCALAR_METRIC_KWARGS)
        batch_metrics = _calc_batch_metrics(ds, config).assign_coords({"batch": i})
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
                print(metric)
    """metric_kwargs = {"weights": [ds["area_weights"]]}
    for var in METRIC_VARS:
        for metric_func in [_bias, _rmse]:
            for comparison in METRIC_COMPARISON_COORDS:
                metric = _calc_metric(ds, metric_func, var, *comparison, metric_kwargs)
                metrics[metric.name] = metric"""
    return metrics


def _insert_additional_variables(ds):
    ds["area_weights"] = ds[AREA_VAR] / (ds[AREA_VAR].mean())
    ds["delp_weights"] = ds[DELP_VAR] / ds[DELP_VAR].mean("z")
    ds["Q1"] = ds["pQ1"] + ds["dQ1"]
    ds["Q2"] = ds["pQ2"] + ds["dQ2"]
    ds = insert_column_integrated_vars(ds, ML_VARS)
    ds = _insert_means(ds, METRIC_VARS, ds["area_weights"])
    return ds


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
    metric = metric_func(da_target, da_predict, **(metric_kwargs or {}))

    metric_name = (
        f"{metric_func.__name__.strip('_')}/{var}/{predict_coord}_vs_{target_coord}"
    )
    return metric.rename(metric_name)


def _insert_means(
    ds: xr.Dataset, vars: Sequence[str], weights: xr.DataArray = None
) -> xr.Dataset:
    for var in vars:
        da = ds[var].sel({DERIVATION_DIM: [TARGET_COORD, PREDICT_COORD]})
        weights = 1.0 if weights is None else weights
        mean_dims = VERTICAL_PROFILE_MEAN_DIMS if "z" in da.dims else None
        mean = (
            (da.sel({DERIVATION_DIM: TARGET_COORD}) * weights)
            .mean(mean_dims)
            .assign_coords({DERIVATION_DIM: "mean"})
        )
        da = xr.concat([da, mean], dim=DERIVATION_DIM)
        ds = ds.drop([var])
        ds = ds.merge(da)
    return ds


def _bias(
    da_target: xr.DataArray,
    da_pred: xr.DataArray,
    weights: xr.DataArray = None,
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
    weights: xr.DataArray = None,
    mean_dims: Sequence[str] = None,
):
    se = (da_target - da_pred) ** 2
    if weights is not None:
        for weight_da in weights:
            se *= weight_da
    return np.sqrt(se.mean(dim=mean_dims))

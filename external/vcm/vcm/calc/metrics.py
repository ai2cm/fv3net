"""Classification and regression scoring routines

Unlike similar sklearn routines these can be used with multi-dimensional or
xarray data easily.

These should mostly work with any array-like data. Most functions take arguments


    truth: a boolean array of true values
    pred: a boolean array of predicted values
    mean: a function used to compute statics. Can be a sample average,
        ``vcm.zonal_average``, ``lambda x: x.mean("time")``, etc.

"""
import functools
import numpy as np
from typing import Union, Sequence, Optional, Callable
import xarray as xr


def r2_score(truth, pred, sample_dim, mean_dims=None):

    if mean_dims is None:
        mean_dims = [sample_dim]
    mean = truth.mean(mean_dims, skipna=True)

    sse = ((truth - pred) ** 2).sum(sample_dim, skipna=True)
    ss = ((truth - mean) ** 2).sum(sample_dim, skipna=True)

    return 1 - sse / ss


def default_mean(x):
    return x.mean()


def assert_bool_args(func):
    @functools.wraps(func)
    def newfunc(truth, pred, mean=default_mean):
        if truth.dtype != bool:
            raise ValueError("dtype {truth.dtype} found expected bool.")

        if pred.dtype != bool:
            raise ValueError("dtype {pred.dtype} found expected bool.")

        return func(truth, pred, mean)

    return newfunc


@assert_bool_args
def accuracy(truth, pred, mean=default_mean):
    """Compute the fraction of correctly classified points::

        TP + TN / (P + N)

    """
    tp = mean(truth & pred)
    tn = mean((~truth) & (~pred))
    return tp + tn


@assert_bool_args
def precision(truth, pred, mean=default_mean):
    """Compute the precision

    Precision (i.e. positive predictive value) is the fraction of predicted
    values that are actually positive::

        TP / (TP + FP)
    """
    tp = mean(truth & pred)
    fp = mean((~truth) & pred)
    return tp / (tp + fp)


@assert_bool_args
def false_positive_rate(truth, pred, mean=default_mean):
    """Compute the false positive rate::

        FP / N

    """
    fp = mean((~truth) & pred)
    n = mean(~truth)
    return fp / n


@assert_bool_args
def true_positive_rate(truth, pred, mean=default_mean):
    """Compute the true positive rate (i.e. recall)::

        TP / P
    """
    tp = mean(truth & pred)
    p = mean(truth)
    return tp / p


@assert_bool_args
def f1_score(truth, pred, mean=default_mean):
    """Compute the f1 score

    F1 is the the harmonic mean of precision and recall, and is often used as a
    score for imbalanced classification problems
    """
    p = precision(truth, pred, mean)
    r = recall(truth, pred, mean)

    return 2 * (p * r) / (p + r)


recall = true_positive_rate


XRData = Union[xr.DataArray, xr.Dataset]


def mean_squared_error(truth, pred, mean=default_mean, **kwargs):
    return mean((truth - pred) ** 2, **kwargs)


def average_over_dims(x: XRData, dims=["x", "y", "tile"]) -> XRData:
    return x.mean(dims)


def zonal_average(
    x: XRData,
    lat: xr.DataArray,
    bins: Optional[Sequence[float]] = None,
    lat_name: str = "lat",
) -> XRData:
    bins = bins or np.arange(-90, 91, 2)
    with xr.set_options(keep_attrs=True):
        output = x.groupby_bins(lat.rename("lat"), bins=bins).mean()
        output = output.rename({"lat_bins": lat_name})
    lats_mid = [lat.item().mid for lat in output[lat_name]]
    return output.assign_coords({lat_name: lats_mid})


def r2(
    target: XRData,
    prediction: XRData,
    mean: Callable[..., XRData] = average_over_dims,
    **mean_func_kwargs
) -> XRData:
    variance = (
        mean(target ** 2, **mean_func_kwargs) - mean(target, **mean_func_kwargs) ** 2
    )
    mse = mean_squared_error(target, prediction, mean=mean, **mean_func_kwargs)
    return 1 - mse / variance

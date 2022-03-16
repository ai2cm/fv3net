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

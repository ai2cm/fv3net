from typing import Callable, Union, Mapping, MutableMapping
from ..._shared import ArrayPacker
import numpy as np
import xarray as xr
import tensorflow as tf


Weight = Union[int, float, np.ndarray]


def _weighted_mse(weights, std):
    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(weights * tf.math.square((y_pred - y_true) / std))

    return custom_loss


def _weighted_mae(weights, std):
    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(weights * tf.math.abs((y_pred - y_true) / std))

    return custom_loss


def _pack_weights(y_packer: ArrayPacker, **weights):
    """Returns a size [1, n_features] array of stacked weights corresponding to a
    stacked array, based on values given in a weights dictionary. Default weight is
    1 split between all features of a quantity. Values can be scalar or an array giving
    a weight for each feature.
    """
    for name in y_packer.pack_names:
        weights[name] = weights.get(name, 1)  # default weight of 1
    _divide_scalar_weights_by_feature_counts(weights, y_packer.feature_counts)
    data_vars = {}
    for name in y_packer.pack_names:
        weight = weights[name]
        if isinstance(weight, np.ndarray):
            array = weight[None, :]
            dims = [y_packer.sample_dim_name, f"{name}_feature"]
        else:
            array = np.zeros([1]) + weight
            dims = [y_packer.sample_dim_name]
        data_vars[name] = (dims, array)
    return y_packer.to_array(xr.Dataset(data_vars))  # type: ignore


def _divide_scalar_weights_by_feature_counts(
    weights: MutableMapping[str, Weight], feature_counts: Mapping[str, int]
):
    for name, total_variable_weight in weights.items():
        if isinstance(total_variable_weight, (int, float)):
            weights[name] = total_variable_weight / feature_counts[name]
        elif isinstance(total_variable_weight, np.ndarray):
            if total_variable_weight.shape == (1,):
                weights[name] = total_variable_weight / feature_counts[name]
        else:
            raise TypeError(
                f"received weight of type {type(total_variable_weight)} for {name}, "
                "only int, float, or ndarray are accepted"
            )


def get_weighted_mae(
    y_packer: ArrayPacker, y_std: np.ndarray, **weights: Weight,
) -> Callable:
    """Retrieve a weighted mean squared error loss function for a given set of weights.

    Each feature is divided by the standard deviation before multiplying by the given
    weights, so that the weight given as input to this function represents the overall
    contribution of that feature to the loss, *independent of how much variance
    that feature has*.

    If you do not want scaling to occur, provide a standard deviation of 1.

    If a uniform weighting is given to specific
    humidity, and the variance of specific humidity is much smaller in upper levels,
    a heavier weight will be given to upper levels so that it contributes just as much
    to the loss as lower levels, despite the lower variance.

    Args:
        y_packer: an object which creates stacked arrays
        y_std: the standard deviation of a stacked array of outputs
        **weights: for each variable, either a scalar weight which will be split between
            all features of that variable, or an array specifying the weight
            for each feature of that variable. Default weight is 1 split between all
            features of that variable. Weights indicate
            how much contribution that feature will give to the loss,
            independent of the variance of that feature.

    Returns:
        weighted_loss
    """
    # convert scalar weights from total variable weight to per-feature weight
    weights = _pack_weights(y_packer, **weights)
    return _weighted_mae(weights, y_std)

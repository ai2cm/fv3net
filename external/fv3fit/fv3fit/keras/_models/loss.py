from typing import Callable, Union
from ..._shared import ArrayPacker
import numpy as np
import xarray as xr
import tensorflow as tf


def _weighted_loss(weights, loss):
    weights = tf.constant(weights.astype(np.float32))

    def custom_loss(y_true, y_pred):
        return loss(tf.multiply(y_true, weights), tf.multiply(y_pred, weights))

    return custom_loss


def _pack_weights(y_packer, y_std, **weights):
    """Returns a size [1, n_features] array of stacked weights corresponding to a
    stacked array, based on values given in a weights dictionary. Default weight is
    1 split between all features of a quantity. Values can be scalar or an array giving
    a weight for each feature.
    """
    data_vars = {}
    for name in y_packer.names:
        weight = weights.get(name, 1)
        if isinstance(weight, np.ndarray):
            array = weight[None, :]
            dims = [y_packer.sample_dim_name, f"{name}_feature"]
        else:
            array = np.zeros([1]) + weight
            dims = [y_packer.sample_dim_name]
        data_vars[name] = (dims, array)
    return y_packer.to_array(xr.Dataset(data_vars)) / y_std


def get_weighted_loss(
    loss: Callable,
    y_packer: ArrayPacker,
    y_std: np.ndarray,
    **weights: Union[int, float, np.ndarray],
) -> Callable:
    """Retrieve a weighted loss function for a given set of weights.

    Args:
        loss: the basic loss function, such as tf.keras.losses.MSE
        y_packer: an object which creates stacked arrays
        y_std: the standard deviation of a stacked array of outputs
        **weights: for each variable, either a scalar weight which will be split between
            all features of that variable, or an array specifying the weight
            for each feature of that variable. Default weight is 1 split between all
            features of that variable.

    Returns:
        weighted_loss
    """
    weights = _pack_weights(y_packer, y_std, **weights)
    return _weighted_loss(weights, loss)

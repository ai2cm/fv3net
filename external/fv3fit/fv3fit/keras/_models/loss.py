from typing import Callable, Union
from ..._shared import ArrayPacker
import numpy as np
import xarray as xr
import tensorflow as tf


def weighted_loss(weights, loss):
    weights = tf.constant(weights.astype(np.float32))

    def custom_loss(y_true, y_pred):
        return loss(tf.multiply(y_true, weights), tf.multiply(y_pred, weights))

    return custom_loss


def get_weighted_loss(
    loss: Callable,
    y_packer: ArrayPacker,
    y_scaler,
    **weights: Union[int, float, np.ndarray],
) -> Callable:
    """Retrieve a weighted loss function for a given set of weights.

    Args:
        loss: the basic loss function, such as tf.keras.losses.MSE
        y_packer: an object which creates stacked arrays
        y_scaler: an object which can normalize or denormalize stacked arrays
        **weights: for each variable, either a scalar weight for all features of that
            variable, or an array specifying the weight for each feature of that
            variable

    Returns:
        weighted_loss
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
    dataset = xr.Dataset(data_vars)
    weights = y_packer.to_array(dataset) / y_scaler.std
    return weighted_loss(weights, loss)

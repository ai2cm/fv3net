import dataclasses
import numpy as np
import tensorflow as tf


def _standard_scaled_mse(std):
    std = tf.constant(std, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.reduce_mean(tf.math.square((y_pred - y_true) / std), axis=0)
        )

    return custom_loss


def _standard_scaled_mae(std):
    std = tf.constant(std, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.reduce_mean(tf.math.abs((y_pred - y_true) / std), axis=0)
        )

    return custom_loss


def _uniform_scaled_mse(std):
    factor = tf.constant(1.0 / np.mean(std ** 2), dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, tf.losses.mse(y_true, y_pred))

    return custom_loss


def _uniform_scaled_mae(std):
    factor = tf.constant(1.0 / np.mean(std), dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, tf.losses.mae(y_true, y_pred))

    return custom_loss


def multiply_loss_by_factor(original_loss, factor):
    def loss(y_true, y_pred):
        return tf.math.scalar_mul(factor, original_loss(y_true, y_pred))

    return loss


@dataclasses.dataclass
class LossConfig:
    """
    Attributes:
        loss_type: one of "mse" or "mae"
        scaling: "standard" corresponds to scaling each feature's lossby
            its scale, "standard_uniform" corresponds to scaling
            each feature's loss by the mean of all feature scales, where
            the scale is variance for MSE loss or standard deviation
            for MAE loss
        weight: A scaling factor by which to modify this loss
    """

    loss_type: str = "mse"
    scaling: str = "standard_uniform"
    weight: float = 1.0

    def __post_init__(self):
        if self.loss_type not in ("mse", "mae"):
            raise ValueError(
                f"loss_type must be 'mse' or 'mae', got '{self.loss_type}'"
            )
        if self.scaling not in ("standard", "standard_uniform"):
            raise ValueError(
                "loss_type must be 'standard' or 'standard_uniform', "
                f"got '{self.scaling}'"
            )

    def loss(self, std: np.ndarray) -> tf.keras.losses.Loss:
        """
        Returns the loss function described by the configuration.

        Args:
            std: standard deviation of the output features
        
        Returns:
            loss: keras loss function
        """
        if self.loss_type == "mse":
            if self.scaling == "standard_uniform":
                loss = _uniform_scaled_mse(std)
            elif self.scaling == "standard":
                loss = _standard_scaled_mse(std)
        elif self.loss_type == "mae":
            if self.scaling == "standard_uniform":
                loss = _uniform_scaled_mae(std)
            elif self.scaling == "standard":
                loss = _standard_scaled_mae(std)
        else:
            raise NotImplementedError(f"loss_type {self.loss_type} is not implemented")
        if self.weight != 1.0:
            loss = multiply_loss_by_factor(loss, self.weight)
        return loss

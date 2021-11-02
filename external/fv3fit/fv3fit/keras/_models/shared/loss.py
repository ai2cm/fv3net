import dataclasses
import numpy as np
import tensorflow as tf


def _standard_scaled_mse(std, ignore_errors_greater_than_stds: float):
    std = tf.constant(std, dtype=std.dtype)
    max_err = tf.constant(ignore_errors_greater_than_stds ** 2, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        # ignore errors greater than n standard deviations
        return tf.math.reduce_mean(
            tf.math.reduce_mean(
                tf.math.minimum(max_err, tf.math.square((y_pred - y_true) / std)),
                axis=0,
            )
        )

    return custom_loss


def _standard_scaled_mae(std, ignore_errors_greater_than_stds: float):
    std = tf.constant(std, dtype=std.dtype)
    max_err = tf.constant(ignore_errors_greater_than_stds, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.reduce_mean(
                tf.math.minimum(max_err, tf.math.abs((y_pred - y_true) / std)), axis=0
            )
        )

    return custom_loss


def _uniform_scaled_mse(std, ignore_errors_greater_than_stds: float):
    factor = tf.constant(1.0 / np.mean(std ** 2), dtype=std.dtype)
    max_err = tf.constant(ignore_errors_greater_than_stds ** 2, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.reduce_mean(
                tf.math.minimum(
                    max_err, tf.math.scalar_mul(factor, tf.math.square(y_pred - y_true))
                ),
                axis=0,
            )
        )

    return custom_loss


def _uniform_scaled_mae(std, ignore_errors_greater_than_stds: float):
    factor = tf.constant(1.0 / np.mean(std), dtype=std.dtype)
    max_err = tf.constant(ignore_errors_greater_than_stds, dtype=std.dtype)

    def custom_loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.reduce_mean(
                tf.math.minimum(
                    max_err, tf.math.scalar_mul(factor, tf.math.abs(y_pred - y_true))
                ),
                axis=0,
            )
        )

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
        ignore_errors_greater_than_stds: errors greater than this number
            of standard deviations will be clipped, and so will not contribute
            to gradient descent
    """

    loss_type: str = "mse"
    scaling: str = "standard_uniform"
    weight: float = 1.0
    ignore_errors_greater_than_stds: float = np.inf

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
        max_error_stds = self.ignore_errors_greater_than_stds
        if self.loss_type == "mse":
            if self.scaling == "standard_uniform":
                loss = _uniform_scaled_mse(
                    std, ignore_errors_greater_than_stds=max_error_stds,
                )
            elif self.scaling == "standard":
                loss = _standard_scaled_mse(
                    std, ignore_errors_greater_than_stds=max_error_stds,
                )
        elif self.loss_type == "mae":
            if self.scaling == "standard_uniform":
                loss = _uniform_scaled_mae(
                    std, ignore_errors_greater_than_stds=max_error_stds,
                )
            elif self.scaling == "standard":
                loss = _standard_scaled_mae(
                    std, ignore_errors_greater_than_stds=max_error_stds,
                )
        else:
            raise NotImplementedError(f"loss_type {self.loss_type} is not implemented")
        if self.weight != 1.0:
            loss = multiply_loss_by_factor(loss, self.weight)
        return loss

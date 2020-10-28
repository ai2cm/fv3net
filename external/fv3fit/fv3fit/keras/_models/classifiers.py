import logging
import tensorflow as tf
import xarray as xr
import numpy as np
from typing import Sequence, Optional, Any, Union, Mapping

from .models import DenseModel
from ._sequences import _XyArraySequence, _TargetToBool, _BalanceNegativeSkewBinary

logger = logging.getLogger(__name__)


class DenseClassifierModel(DenseModel):
    def get_model(
        self, n_features_in: int, n_features_out: int, weights=None
    ) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            x = tf.keras.layers.Dense(
                self._width, activation=tf.keras.activations.relu
            )(x)
        outputs = tf.keras.layers.Dense(n_features_out)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self._optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
            loss_weights=weights,
        )
        return model

    def fit(
        self,
        batches: Sequence[xr.Dataset],
        epochs: int = 1,
        batch_size: Optional[int] = None,
        workers: int = 1,
        max_queue_size: int = 8,
        loss_weights: Any = None,
        true_threshold: Union[float, int, np.ndarray] = None,
        balance_samples: bool = False,
        **fit_kwargs: Any,
    ) -> None:

        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)

        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            if loss_weights == "rms":
                rms = np.sqrt((y ** 2).mean(axis=0))
                wgt = rms / rms.sum()
                self._loss_weights = [wgt]
            else:
                self._loss_weights = loss_weights

            self._model = self.get_model(
                n_features_in, n_features_out, weights=self._loss_weights
            )

        Xy = _TargetToBool(self.X_packer, self.y_packer, batches)
        default_thresh = self.y_scaler.std.max() * 10 ** -4
        thresh = true_threshold if true_threshold is not None else default_thresh
        Xy.set_y_thresh(thresh)

        if balance_samples:
            Xy = _BalanceNegativeSkewBinary(Xy)

        if batch_size is not None:
            self._fit_loop(
                Xy,
                epochs,
                batch_size,
                workers=workers,
                max_queue_size=max_queue_size,
                **fit_kwargs,
            )
        else:
            self._fit_array(
                Xy,
                epochs=epochs,
                workers=workers,
                max_queue_size=max_queue_size,
                **fit_kwargs,
            )


def get_feature_slices(packer):

    slices = {}
    start = 0
    for pack_name in packer.pack_names:
        n_features = packer.feature_counts[pack_name]
        slices[pack_name] = slice(start, start + n_features)
        start += n_features

    return slices


def _get_mask(x):

    non_zero = tf.math.greater_equal(x, tf.zeros_like(x))
    mask = tf.where(non_zero, x=tf.ones_like(x), y=tf.zeros_like(x))

    return mask


def separate_tensor_by_var(tensor, feature_dim_slices):

    # assumes sample dim leading
    sep = {
        var_name: tensor[slice(None), feature_dim_slices[var_name]]
        for var_name in feature_dim_slices
    }

    return sep


class DenseWithClassifier(DenseModel):
    def __init__(
        self, *args, classifiers: Mapping[str, DenseClassifierModel] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if classifiers is None:
            self._classifiers = {}
            logger.warning("DenseWithClassifier initiwalized without any classiefiers")
        else:
            self._classifiers = classifiers

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:

        out_tensor_slices = get_feature_slices(self.y_packer)

        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            x = tf.keras.layers.Dense(
                self._width, activation=tf.keras.activations.relu
            )(x)
        x = tf.keras.layers.Dense(n_features_out)(x)
        regr_outputs = self.y_scaler.denormalize_layer(x)

        regr_outputs_by_var = separate_tensor_by_var(regr_outputs, out_tensor_slices)

        to_combine = []
        for var, out_tensor in regr_outputs_by_var.items():
            if var in self._classifiers:
                classified = self._classifiers[var]._model(inputs)
                mask = tf.keras.layers.Lambda(_get_mask)(classified)
                filtered = tf.keras.layers.Multiply()([out_tensor, mask])
                to_combine.append(filtered)
            else:
                to_combine.append(out_tensor)

        output = tf.concat(to_combine, 1)

        model = tf.keras.Model(
            inputs=inputs, outputs=output, name=self.__class__.__name__
        )
        model.compile(
            optimizer=self._optimizer,
            loss=self.loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )
        return model

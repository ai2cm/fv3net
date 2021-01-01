import logging
import tensorflow as tf
import xarray as xr
import numpy as np
from typing import Sequence, Optional, Any, Union, Mapping

from .models import DenseModel
from ._sequences import _XyArraySequence, _TargetToBool, _BalanceNegativeSkewBinary

logger = logging.getLogger(__name__)


class DenseClassifierModel(DenseModel):

    def __init__(self,*args, true_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = true_threshold

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

    def _get_options(self):
        options = super()._get_options()
        options["true_threshold"] = self.threshold
        return options

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

            default_thresh = self.y_scaler.std.max() * 10 ** -4
            self.threshold = (
                true_threshold if true_threshold is not None else default_thresh
            )

        Xy = _TargetToBool(self.X_packer, self.y_packer, batches)
        Xy.set_y_thresh(self.threshold)

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


def separate_tensor_by_var(tensor, feature_dim_slices):

    # assumes sample dim leading
    sep = {
        var_name: tensor[slice(None), feature_dim_slices[var_name]]
        for var_name in feature_dim_slices
    }

    return sep


def _less_than_zero_mask(x):
    return tf.math.greater_equal(x, tf.constant(0, dtype=x.dtype))


def _less_than_equal_zero_mask(x):
    return tf.math.greater(x, tf.constant(0, dtype=x.dtype))


def _empty_column_sample_mask(x):

    tracer_column_integral = tf.math.reduce_sum(x, axis=1, keepdims=True)
    mask = tf.math.greater(tracer_column_integral, tf.constant(0, dtype=x.dtype))

    return mask


class DenseWithClassifier(DenseModel):
    def __init__(
        self,
        *args,
        classifiers: Mapping[str, DenseClassifierModel] = None,
        mask_with_classifier: Sequence[str] = None,
        limit_zero: Sequence[str] = None,
        convert_int: Sequence[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if classifiers is None:
            classifiers = {}
            logger.warning("DenseWithClassifier initialized without any classiefiers")
        self._classifiers = classifiers

        if mask_with_classifier is None:
            mask_with_classifier = []
            logger.warning("DenseWithClassifier not applying mask to any vars.")
        self._mask_with_classifier = mask_with_classifier

        self._limit_zero = limit_zero if limit_zero is not None else []
        self._convert_int = convert_int if convert_int is not None else []

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:

        in_tensor_slices = get_feature_slices(self.X_packer)
        out_tensor_slices = get_feature_slices(self.y_packer)

        inputs = tf.keras.Input(n_features_in)
        inputs_by_var = separate_tensor_by_var(inputs, in_tensor_slices)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            x = tf.keras.layers.Dense(
                self._width, activation=tf.keras.activations.relu
            )(x)
        x = tf.keras.layers.Dense(n_features_out)(x)
        regr_outputs = self.y_scaler.denormalize_layer(x)

        regr_outputs_by_var = separate_tensor_by_var(regr_outputs, out_tensor_slices)
        
        dtype = regr_outputs.dtype
        zero = tf.constant(0, dtype=dtype)
        # Create a mask from combining all classifiers
        classifier_ens = []
        for classifier in self._classifiers:
            classified = classifier._model(inputs)
            mask = tf.math.greater_equal(classified, zero)
            classifier_ens.append(mask)
        
        combo_mask = classifier_ens[0]
        for mask in classifier_ens[1:]:
            combo_mask = tf.logical_or(combo_mask, mask)

        top_zero = np.ones((combo_mask.shape[-1]), dtype=np.bool)
        top_zero[-10:] = 0
        top_zero = tf.convert_to_tensor(top_zero)
        combo_mask = tf.logical_and(combo_mask, top_zero)

        to_combine = []
        for var, out_tensor in regr_outputs_by_var.items():
            dtype = out_tensor.dtype

            if var in self._mask_with_classifier:
                out_tensor = tf.where(combo_mask, x=out_tensor, y=zero)

            if var in self._limit_zero:
                mask = tf.math.greater_equal(out_tensor, zero)
                out_tensor = tf.where(mask, x=out_tensor, y=zero)

            if var in self._convert_int:
                out_tensor = tf.keras.layers.Lambda(lambda x: tf.math.round(x))(
                    out_tensor
                )

            # Zero out samples without anything in the column
            if "rtg_output" in var:
                # look for other tracer fields
                tracer_num = var.split("_")[-1]
                q1_name = f"q1_input_{tracer_num}"
                if q1_name in inputs_by_var:
                    logger.debug(f"Found matching rtg input to use for mask: {q1_name}")
                    q1 = inputs_by_var[q1_name]
                    tracer_column_integral = tf.math.reduce_sum(q1, axis=1, keepdims=True)
                    mask = tf.math.greater(tracer_column_integral, zero)
                    out_tensor = tf.where(mask, x=out_tensor, y=zero)

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

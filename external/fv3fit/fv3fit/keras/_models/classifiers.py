import tensorflow as tf
import xarray as xr
from typing import Sequence, Optional, Any, Union

from .models import DenseModel, DenseWithClassifier
from ._sequences import _XyArraySequence, _TargetToBool


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
            loss_weights=weights
        )
        return model
    
    def fit(
        self,
        batches: Sequence[xr.Dataset],
        epochs: int = 1,
        batch_size: Optional[int] = None,
        workers: int = 1,
        max_queue_size: int = 8,
        loss_weights: Sequence[Any] = None,
        true_threshold: Union[float, int] = None,
        **fit_kwargs: Any,
    ) -> None:
        
        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)

        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            self._model = self.get_model(n_features_in, n_features_out, weights=loss_weights)
            
        Xy = _TargetToBool(self.X_packer, self.y_packer, batches)
        default_thresh = self.y_scaler.std * 10 ** -4
        thresh = true_threshold or default_thresh if true_threshold is None
        Xy.set_y_thresh(self.y_scaler.std * 10 ** -4)

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

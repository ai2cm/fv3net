from typing import Iterable, Sequence
import logging
import os
import xarray as xr
import numpy as np
import tensorflow as tf
from .models import Model, _XyArraySequence
from ._filesystem import get_dir, put_dir
from ..._shared import ArrayPacker

logger = logging.getLogger(__file__)


class DummyModel(Model):
    """
    A dummy keras model for testing, whose `fit` method learns only the input and
    output variable array dimensions in an xarray dataset and ignores their contents,
    and which simply returns zeros for all output variable features
    """

    # these should only be used in the dump/load routines for this class
    _MODEL_FILENAME = "model.tf"
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        """Initialize the DummyModel
        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=output_variables
        )

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def fit(self, batches: Sequence[xr.Dataset]) -> None:
        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)
        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._model = self.get_model(n_features_in, n_features_out)

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        outputs = tf.keras.layers.Lambda(lambda x: x * 0.0)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile()
        return model

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        feature_index = X[self.sample_dim_name]
        ds_pred = self.y_packer.to_dataset(
            self.predict_array(self.X_packer.to_array(X))
        )
        return ds_pred.assign_coords({self.sample_dim_name: feature_index})

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self._model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)

    @classmethod
    def load(cls, path: str) -> Model:
        with get_dir(path) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            obj = cls(
                X_packer.sample_dim_name, X_packer.pack_names, y_packer.pack_names
            )
            obj.X_packer = X_packer
            obj.y_packer = y_packer

            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                obj._model = tf.keras.models.load_model(model_filename)
            return obj

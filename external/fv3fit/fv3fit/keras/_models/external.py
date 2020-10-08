from typing import Iterable, Hashable, Dict, Any
import tensorflow as tf
import os
from ..._shared import Predictor, ArrayPacker
from ._filesystem import get_dir, put_dir
import xarray as xr


class ExternalModel(Predictor):
    """Model initialized using pre-trained objects."""

    _MODEL_FILENAME = "model.tf"
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: tf.keras.Model,
        X_packer: ArrayPacker,
        y_packer: ArrayPacker,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """
        if X_packer.sample_dim_name != sample_dim_name:
            raise ValueError(
                f"must provide sample_dim_name compatible with "
                "X_packer.sample_dim_name, got "
                f"{sample_dim_name} and {X_packer.sample_dim_name}"
            )
        if y_packer.sample_dim_name != sample_dim_name:
            raise ValueError(
                f"must provide sample_dim_name compatible with "
                "y_packer.sample_dim_name, got "
                f"{sample_dim_name} and {y_packer.sample_dim_name}"
            )
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.sample_dim_name = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self.X_packer = X_packer
        self.y_packer = y_packer

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        sample_coord = X[self.sample_dim_name]
        ds_pred = self.y_packer.to_dataset(
            self.model.predict(self.X_packer.to_array(X))
        )
        return ds_pred.assign_coords({self.sample_dim_name: sample_coord})

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)

    @classmethod
    def load(cls, path: str) -> "ExternalModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                model = tf.keras.models.load_model(
                    model_filename, custom_objects=cls.custom_objects
                )
            obj = cls(
                X_packer.sample_dim_name,
                X_packer.pack_names,
                y_packer.pack_names,
                model,
                X_packer,
                y_packer,
            )
            return obj

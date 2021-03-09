from typing import Dict, Any, Iterable, Hashable
import tensorflow as tf
from .gcm_cell import GCMCell
import xarray as xr
import os
from ..._shared import Predictor, ArrayPacker, StandardScaler, io
from ._filesystem import get_dir, put_dir
import xarray as xr
import copy
import yaml


class ExternalModel(Predictor):
    """Model initialized using pre-trained objects."""

    _MODEL_FILENAME = "model.tf"
    _INPUT_PACKER_FILENAME = "input_packer.json"
    _PROGNOSTIC_PACKER_FILENAME = "prognostic_packer.json"
    _INPUT_SCALER_FILENAME = "input_scaler"
    _PROGNOSTIC_SCALER_FILENAME = "prognostic_scaler"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: tf.keras.Model,
        input_packer: ArrayPacker,
        prognostic_packer: ArrayPacker,
        # IO only set up for StandardScaler right now, need to save class info to
        # support other scaler classes
        input_scaler: StandardScaler,
        prognostic_scaler: StandardScaler,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """
        if input_packer.sample_dim_name != sample_dim_name:
            raise ValueError(
                f"must provide sample_dim_name compatible with "
                "X_packer.sample_dim_name, got "
                f"{sample_dim_name} and {input_packer.sample_dim_name}"
            )
        if prognostic_packer.sample_dim_name != sample_dim_name:
            raise ValueError(
                f"must provide sample_dim_name compatible with "
                "y_packer.sample_dim_name, got "
                f"{sample_dim_name} and {prognostic_packer.sample_dim_name}"
            )
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.sample_dim_name = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model
        self.input_packer = input_packer
        self.prognostic_packer = prognostic_packer
        self.input_scaler = input_scaler
        self.prognostic_scaler = prognostic_scaler

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
            with open(os.path.join(path, self._INPUT_PACKER_FILENAME), "w") as f:
                self.input_packer.dump(f)
            with open(os.path.join(path, self._PROGNOSTIC_PACKER_FILENAME), "w") as f:
                self.prognostic_packer.dump(f)
            with open(os.path.join(path, self._INPUT_SCALER_FILENAME), "wb") as f:
                self.input_scaler.dump(f)
            with open(os.path.join(path, self._PROGNOSTIC_SCALER_FILENAME), "wb") as f:
                self.prognostic_scaler.dump(f)

    @classmethod
    def load(cls, path: str) -> "ExternalModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            with open(os.path.join(path, cls._INPUT_PACKER_FILENAME), "r") as f:
                input_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._PROGNOSTIC_PACKER_FILENAME), "r") as f:
                prognostic_packer = ArrayPacker.load(f)
            # currently only supports standard scaler, need to change dump and load
            # to save scaler type to support other scalers
            with open(os.path.join(path, cls._INPUT_SCALER_FILENAME), "r") as f:
                input_scaler = StandardScaler.load(f)
            with open(os.path.join(path, cls._PROGNOSTIC_SCALER_FILENAME), "r") as f:
                prognostic_scaler = StandardScaler.load(f)
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                model = tf.keras.models.load_model(
                    model_filename, custom_objects=cls.custom_objects
                )
            obj = cls(
                input_packer.sample_dim_name,
                input_packer.pack_names,
                prognostic_packer.pack_names,
                model,
                input_packer,
                prognostic_packer,
                input_scaler,
                prognostic_scaler,
            )
            return obj


@io.register("recurrent-keras")
class RecurrentModel(ExternalModel):

    _CONFIG_FILENAME = "recurrent_model.yaml"
    custom_objects: Dict[str, Any] = {
        "custom_loss": tf.keras.losses.mse,
        "GCMCell": GCMCell,
    }

    def __init__(self, 
            sample_dim_name: str,
            input_variables: Iterable[Hashable],
            model: tf.keras.Model,
            input_packer: ArrayPacker,
            prognostic_packer: ArrayPacker,
            input_scaler: StandardScaler,
            prognostic_scaler: StandardScaler,
            train_timestep_seconds: int,
        ):
        output_variables = ["dQ1", "dQ2"]
        input_variables = list(input_variables) + ["air_temperature", "specific_humidity"]
        super(RecurrentModel, self).__init__(
            sample_dim_name,
            input_variables,
            output_variables,
            model,
            input_packer,
            prognostic_packer,
            input_scaler,
            prognostic_scaler
        )
        self.train_timestep_seconds = train_timestep_seconds
        self.tendency_scaler = copy.deepcopy(prognostic_scaler)
        self.tendency_scaler.mean[:] = 0.  # don't remove mean for tendencies

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        sample_coord = X[self.sample_dim_name]
        forcing = self.input_packer.to_array(X)
        state_in = self.prognostic_packer.to_array(X)
        norm_forcing = self.input_scaler.normalize(forcing)
        norm_state_in = self.prognostic_scaler.normalize(state_in)
        norm_state_out = self.model.predict([norm_forcing, norm_state_in])
        tendency_out = self.tendency_scaler.denormalize(
            norm_state_out - norm_state_in
        ) / self.train_timestep_seconds # divide difference by timestep to get s^-1 units
        # assert False
        ds_tendency = self.prognostic_packer.to_dataset(tendency_out)
        ds_pred = xr.Dataset({})
        ds_pred["dQ1"] = ds_tendency["air_temperature"]
        ds_pred["dQ2"] = ds_tendency["specific_humidity"]
        return ds_pred.assign_coords({self.sample_dim_name: sample_coord})

    def dump(self, path: str) -> None:
        super().dump(path)
        with put_dir(path) as path:
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.safe_dump(
                        {"train_timestep_seconds": self.train_timestep_seconds}
                    )
                )

    @classmethod
    def load(cls, path: str) -> "RecurrentModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            with open(os.path.join(path, cls._INPUT_PACKER_FILENAME), "r") as f:
                input_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._PROGNOSTIC_PACKER_FILENAME), "r") as f:
                prognostic_packer = ArrayPacker.load(f)
            # currently only supports standard scaler, need to change dump and load
            # to save scaler type to support other scalers
            with open(os.path.join(path, cls._INPUT_SCALER_FILENAME), "rb") as f:
                input_scaler = StandardScaler.load(f)
            with open(os.path.join(path, cls._PROGNOSTIC_SCALER_FILENAME), "rb") as f:
                prognostic_scaler = StandardScaler.load(f)
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                model = tf.keras.models.load_model(
                    model_filename, custom_objects=cls.custom_objects
                )
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                train_timestep_seconds = yaml.safe_load(f)["train_timestep_seconds"]
            obj = cls(
                input_packer.sample_dim_name,
                input_packer.pack_names,
                model,
                input_packer,
                prognostic_packer,
                input_scaler,
                prognostic_scaler,
                train_timestep_seconds
            )
            return obj

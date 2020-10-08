from typing import Dict, Any
import tensorflow as tf
from .gcm_cell import GCMCell
from .external import ExternalModel
import xarray as xr


class RecurrentModel(ExternalModel):

    custom_objects: Dict[str, Any] = {
        "custom_loss": tf.keras.losses.mse,
        "GCMCell": GCMCell,
    }

    def __init__(self, *args, **kwargs):
        super(RecurrentModel, self).__init__(*args, **kwargs)
        required_outputs = ["air_temperature", "specific_humidity"]
        if any(name not in self.output_variables for name in required_outputs):
            raise ValueError(f"output variables must include all of {required_outputs}")
        self.output_variables = tuple(list(self.output_variables) + ["dQ1", "dQ2"])
        self.set_mode(lock_to_inputs=True)

    def set_mode(self, *, lock_to_inputs: bool):
        """Set whether the RecurrentModel internal state is allowed to
        evolve independently of the input values, or should be set
        exactly to the input values.

        If the internal state is allowed to evolve independently,
        only the difference between successive inputs is added to the GCM state.
        """
        cell = self.model.get_layer("rnn").cell
        cell.lock_to_inputs = lock_to_inputs

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        sample_coord = X[self.sample_dim_name]
        forcing = self.X_packer.to_array(X)
        state_in = self.y_packer.to_array(X)
        state_out = self.model.predict([forcing, state_in])
        ds_tendency = self.y_packer.to_dataset(state_out - state_in)
        ds_pred = self.y_packer.to_dataset(state_out)
        ds_pred["dQ1"] = ds_tendency["specific_humidity"] / (
            60 * 60
        )  # 1 hour timestep, to s^-1
        ds_pred["dQ2"] = ds_tendency["air_temperature"] / (60 * 60)
        return ds_pred.assign_coords({self.sample_dim_name: sample_coord})

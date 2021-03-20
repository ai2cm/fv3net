from typing import Dict, Any, Iterable, Hashable, Tuple, Optional, Mapping
import tensorflow as tf
from .gcm_cell import GCMCell
import xarray as xr
import os
from ..._shared import Predictor, ArrayPacker, io, StandardScaler
from ._filesystem import get_dir, put_dir
from .normalizer import LayerStandardScaler
import copy
import yaml
import numpy as np


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

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._INPUT_PACKER_FILENAME), "w") as f:
                self.input_packer.dump(f)
            with open(os.path.join(path, self._PROGNOSTIC_PACKER_FILENAME), "w") as f:
                self.prognostic_packer.dump(f)
            with open(os.path.join(path, self._INPUT_SCALER_FILENAME), "wb") as f_bin:
                self.input_scaler.dump(f_bin)
            with open(
                os.path.join(path, self._PROGNOSTIC_SCALER_FILENAME), "wb"
            ) as f_bin:
                self.prognostic_scaler.dump(f_bin)

    # @classmethod
    # def load(cls, path: str) -> "ExternalModel":
    #     """Load a serialized model from a directory."""
    #     with get_dir(path) as path:
    #         with open(os.path.join(path, cls._INPUT_PACKER_FILENAME), "r") as f:
    #             input_packer = ArrayPacker.load(f)
    #         with open(os.path.join(path, cls._PROGNOSTIC_PACKER_FILENAME), "r") as f:
    #             prognostic_packer = ArrayPacker.load(f)
    #         # currently only supports standard scaler, need to change dump and load
    #         # to save scaler type to support other scalers
    #         with open(os.path.join(path, cls._INPUT_SCALER_FILENAME), "r") as f:
    #             input_scaler = StandardScaler.load(f)
    #         with open(os.path.join(path, cls._PROGNOSTIC_SCALER_FILENAME), "r") as f:
    #             prognostic_scaler = StandardScaler.load(f)
    #         model_filename = os.path.join(path, cls._MODEL_FILENAME)
    #         if os.path.exists(model_filename):
    #             model = tf.keras.models.load_model(
    #                 model_filename, custom_objects=cls.custom_objects
    #             )
    #         obj = cls(
    #             input_packer.sample_dim_name,
    #             input_packer.pack_names,
    #             prognostic_packer.pack_names,
    #             model,
    #             input_packer,
    #             prognostic_packer,
    #             input_scaler,
    #             prognostic_scaler,
    #         )
    #         return obj


@io.register("recurrent-keras")
class RecurrentModel(ExternalModel):

    _CONFIG_FILENAME = "recurrent_model.yaml"
    custom_objects: Dict[str, Any] = {
        "custom_loss": tf.keras.losses.mse,
        "GCMCell": GCMCell,
    }

    def __init__(
        self,
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
        input_variables = list(input_variables) + [
            "air_temperature",
            "specific_humidity",
        ]
        super(RecurrentModel, self).__init__(
            sample_dim_name,
            input_variables,
            output_variables,
            model,
            input_packer,
            prognostic_packer,
            input_scaler,
            prognostic_scaler,
        )
        self.train_timestep_seconds = train_timestep_seconds
        self.tendency_scaler = copy.deepcopy(prognostic_scaler)
        self.tendency_scaler.mean[:] = 0.0  # don't remove mean for tendencies

    def _get_inputs(self, X: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        forcing = self.input_packer.to_array(X)
        state_in = self.prognostic_packer.to_array(X)
        norm_forcing = self.input_scaler.normalize(forcing)
        norm_state_in = self.prognostic_scaler.normalize(state_in)
        return norm_forcing, norm_state_in

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        if self.sample_dim_name in X:
            sample_coord = X[self.sample_dim_name]
        else:
            sample_coord = None
        norm_forcing, norm_state_in = self._get_inputs(X)
        norm_state_out = self.model.predict([norm_forcing, norm_state_in])
        tendency_out = (
            self.tendency_scaler.denormalize(norm_state_out - norm_state_in)
            / self.train_timestep_seconds
        )  # divide difference by timestep to get s^-1 units
        # assert False
        ds_tendency = self.prognostic_packer.to_dataset(tendency_out)
        ds_pred = xr.Dataset({})
        ds_pred["dQ1"] = ds_tendency["air_temperature"]
        ds_pred["dQ2"] = ds_tendency["specific_humidity"]
        if sample_coord is not None:
            ds_pred = ds_pred.assign_coords({self.sample_dim_name: sample_coord})
        return ds_pred

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
            with open(os.path.join(path, cls._INPUT_SCALER_FILENAME), "rb") as f_bin:
                input_scaler = StandardScaler.load(f_bin)
            with open(
                os.path.join(path, cls._PROGNOSTIC_SCALER_FILENAME), "rb"
            ) as f_bin:
                prognostic_scaler = StandardScaler.load(f_bin)
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
                train_timestep_seconds,
            )
            return obj


class BPTTModel(Predictor):

    TIME_DIM_NAME = "time"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        n_units: int,
        n_hidden_layers: int,
        # tendency_ratio: float,
        kernel_regularizer,
        train_batch_size: int = 512,
        optimizer="adam",
    ):
        prognostic_variables = (
            "air_temperature",
            "specific_humidity",
        )
        self.given_tendency_names = (
            "air_temperature_tendency_due_to_model",
            "specific_humidity_tendency_due_to_model",
        )
        output_variables = ["dQ1", "dQ2"]
        super(BPTTModel, self).__init__(
            sample_dim_name, input_variables, output_variables,
        )
        self.input_packer = ArrayPacker(sample_dim_name, input_variables)
        self.prognostic_packer = ArrayPacker(sample_dim_name, prognostic_variables)
        self.train_timestep_seconds: Optional[float] = None
        self.input_scaler = LayerStandardScaler()
        self.prognostic_scaler = LayerStandardScaler()
        self.tendency_scaler = LayerStandardScaler(fit_mean=False)
        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        # self.tendency_ratio = tendency_ratio
        self.kernel_regularizer = kernel_regularizer
        self.train_batch_size = train_batch_size
        self.optimizer = optimizer
        self.activation = "relu"

    def build_for(self, X: xr.Dataset):
        """
        Given a dataset with [sample, time, z] and [sample, time] arrays
        containing data in sample windows, fit the scalers and packers
        and define the keras model (e.g. with the given window length).
        """
        inputs = self.input_packer.to_array(X, force_3d=True)
        state = self.prognostic_packer.to_array(X, force_3d=True)
        self.input_scaler.fit(inputs)
        self.prognostic_scaler.fit(state)
        self.tendency_scaler.std = self.prognostic_scaler.std
        self.tendency_scaler.mean = np.zeros_like(self.prognostic_scaler.mean)
        time = X[BPTTModel.TIME_DIM_NAME]
        self._train_timestep_seconds = (
            time[1].values.item() - time[0].values.item()
        ).total_seconds()
        self.train_model, self.predict_model = self.build(len(time) - 1)

    @property
    def losses(self):
        assert tuple(self.prognostic_packer.pack_names) == (
            "air_temperature",
            "specific_humidity",
        )
        std = self.prognostic_packer.to_dataset(self.prognostic_scaler.std[None, :])

        air_temperature_factor = tf.constant(
            0.5 / np.mean(std["air_temperature"].values) ** 2, dtype=tf.float32
        )
        specific_humidity_factor = tf.constant(
            0.5 / np.mean(std["specific_humidity"].values) ** 2, dtype=tf.float32
        )

        # air_temperature_factor = tf.constant(
        #     1.0, dtype=tf.float32
        # )
        # specific_humidity_factor = tf.constant(
        #     1e3, dtype=tf.float32
        # )

        def air_temperature_loss(y_true, y_pred):
            return tf.math.multiply(
                air_temperature_factor, tf.losses.mse(y_true, y_pred)
            )

        def specific_humidity_loss(y_true, y_pred):
            return tf.math.multiply(
                specific_humidity_factor, tf.losses.mse(y_true, y_pred)
            )

        return air_temperature_loss, specific_humidity_loss



    def build(self, n_window):
        def get_vector(packer, scaler, series=True):
            features = [
                packer.feature_counts[name]
                for name in packer.pack_names
            ]
            if series:
                input_layers = [
                    tf.keras.layers.Input(shape=[n_window, n_features])
                    for n_features in features
                ]
            else:
                input_layers = [
                    tf.keras.layers.Input(shape=[n_features])
                    for n_features in features
                ]
            input_vector = scaler.normalize_layer(
                packer.pack_layer()(input_layers)
            )
            return input_layers, input_vector
        input_series_layers, forcing_series_input = get_vector(
            self.input_packer, self.input_scaler, series=True
        )
        state_layers, state_input = get_vector(
            self.prognostic_packer, self.prognostic_scaler, series=False
        )
        given_tendency_series_layers, given_tendency_series_input = get_vector(
            self.prognostic_packer, self.tendency_scaler, series=True
        )
        
        def prepend_forcings(state, i):
            select = tf.keras.layers.Lambda(
                lambda x: x[:, i, :], name=f"select_forcing_input_{i}"
            )
            forcing_input = select(forcing_series_input)
            return tf.keras.layers.concatenate([forcing_input, state])

        def get_given_tendency(i):
            select = tf.keras.layers.Lambda(
                lambda x: x[:, i, :], name=f"select_given_tendency_{i}"
            )
            return select(given_tendency_series_input)

        dense_layers = [
            tf.keras.layers.Dense(
                self.n_units,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}",
            )
            for i in range(self.n_hidden_layers)
        ]
        output_layer = tf.keras.layers.Dense(state_input.shape[-1])
        train_timestep = np.asarray(self._train_timestep_seconds, dtype=np.float32)
        timestep_divide_layer = tf.keras.layers.Lambda(
            lambda x: x / train_timestep, name="timestep_divide",
        )

        def get_predicted_tendency(x):
            for layer in dense_layers:
                x = layer(x)
            return timestep_divide_layer(output_layer(x))
            # return output_layer(x)

        tendency_add_layer = tf.keras.layers.Add(name="tendency_add")
        state_add_layer = tf.keras.layers.Add(name="state_add")
        timestep_multiply_layer = tf.keras.layers.Lambda(
            lambda x: x * train_timestep, name="timestep_multiply",
        )
        add_time_dim_layer = tf.keras.layers.Lambda(lambda x: x[:, None, :])

        state_outputs_list = []
        state = state_input
        for i in range(n_window):
            x = prepend_forcings(state, i)
            predicted_tendency = get_predicted_tendency(x)
            given_tendency = get_given_tendency(i)
            total_tendency = tendency_add_layer([predicted_tendency, given_tendency])
            state_update = timestep_multiply_layer(total_tendency)
            state = state_add_layer([state, state_update])
            state_outputs_list.append(add_time_dim_layer(state))

        norm_state_output = tf.keras.layers.concatenate(state_outputs_list, axis=1)
        state_outputs = self.prognostic_packer.unpack_layer(feature_dim=2)(
            self.prognostic_scaler.denormalize_layer(norm_state_output)
        )

        train_model = tf.keras.Model(
            inputs=input_series_layers + state_layers + given_tendency_series_layers,
            outputs=state_outputs,
        )

        features = [
            self.input_packer.feature_counts[name]
            for name in self.input_packer.pack_names
        ]
        input_layers = [
            tf.keras.layers.Input(shape=[n_features]) for n_features in features
        ]
        features = [
            self.prognostic_packer.feature_counts[name]
            for name in self.prognostic_packer.pack_names
        ]
        state_layers = [
            tf.keras.layers.Input(shape=[n_features]) for n_features in features
        ]
        forcing_input = self.input_scaler.normalize_layer(
            self.input_packer.pack_layer()(input_layers)
        )
        state_input = self.prognostic_scaler.normalize_layer(
            self.prognostic_packer.pack_layer()(state_layers)
        )
        x = tf.keras.layers.concatenate([forcing_input, state_input])
        predicted_tendency = get_predicted_tendency(x)

        tendency_outputs = self.prognostic_packer.unpack_layer(feature_dim=1)(
            self.tendency_scaler.denormalize_layer(predicted_tendency)
        )

        predict_model = tf.keras.Model(
            inputs=input_layers + state_layers, outputs=tendency_outputs
        )

        train_model.compile(optimizer=self.optimizer, loss=self.losses)
        predict_model.compile(optimizer=self.optimizer, loss=tf.keras.losses.mse)

        return train_model, predict_model

    def fit(self, X: xr.Dataset, *, epochs=1):
        """
        Args:
            X: training data
            epochs: how many epochs to train. default is 1 for the case when X
                contains one batch at a time
            validation: if given, output validation loss at end of training
        """
        if self.train_model is None:
            raise RuntimeError("Must call `build_for` method before calling fit")
        if tuple(self.prognostic_packer.pack_names) != (
            "air_temperature",
            "specific_humidity",
        ):
            raise ValueError(
                "given tendency is hard-coded to use air_temperature and "
                "specific_humidity as prognostic names, if the prognostic "
                "packer does not use them there must be a bug "
                "or this check should be removed. "
                f"got names {self.prognostic_packer.pack_names}"
            )

        for name in list(self.input_packer.pack_names) + list(
            self.prognostic_packer.pack_names
        ):
            assert X[name].dims[1] == BPTTModel.TIME_DIM_NAME

        self.train_model.fit(
            x=self.get_keras_inputs(X),
            y=self.get_target_state(X),
            batch_size=self.train_batch_size,
            epochs=epochs,
            shuffle=True,
        )

    def get_keras_inputs(self, X):
        def get_inputs(X):
            return get_keras_arrays(X, self.input_packer.pack_names, slice(0, -1))

        def get_initial_state(X):
            return get_keras_arrays(X, self.prognostic_packer.pack_names, 0)

        def get_given_tendency(X):
            return get_keras_arrays(X, self.given_tendency_names, slice(0, -1))
        return get_inputs(X) + get_initial_state(X) + get_given_tendency(X)


    def get_target_state(self, X):
        return get_keras_arrays(
            X, self.prognostic_packer.pack_names, slice(1, None)
        )

    @classmethod
    def load(cls, path):
        raise NotImplementedError(
            "class meant for training only, should be using AllKerasModel to load/predict"
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "class meant for training only, should be using AllKerasModel to load/predict"
        )

    @property
    def predictor_model(self):
        predictor_model = AllKerasModel(
            self.sample_dim_name,
            tuple(self.input_packer.pack_names)
            + tuple(self.prognostic_packer.pack_names),
            self.output_variables,
            self.predict_model,
        )
        return predictor_model

    def dump(self, path: str) -> None:
        raise NotImplementedError(
            "class meant for training only, dump `predictor_model` attribute instead"
        )


def get_keras_arrays(X, names, time_index):
    return_list = []
    for name in names:
        if len(X[name].dims) == 3:
            return_list.append(X[name].values[:, time_index, :])
        elif len(X[name].dims) == 2:
            return_list.append(X[name].values[:, time_index, None])
        else:
            raise ValueError(X[name].shape)
    return return_list


@io.register("all-keras")
class AllKerasModel(Predictor):
    """Model which uses Keras for packing and normalization"""

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: tf.keras.Model,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.sample_dim_name = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model = model

    @classmethod
    def load(cls, path: str) -> "AllKerasModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            model = tf.keras.models.load_model(
                model_filename, custom_objects=cls.custom_objects
            )
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.safe_load(f)
            obj = cls(
                config["sample_dim_name"],
                config["input_variables"],
                config["output_variables"],
                model,
            )
            return obj

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        inputs = [X[name].values for name in self.input_variables]
        dQ1, dQ2 = self.model.predict(inputs)
        return xr.Dataset(
            data_vars={
                "dQ1": xr.DataArray(
                    dQ1,
                    dims=X["air_temperature"].dims,
                    coords=X["air_temperature"].coords,
                    attrs={"units": X["air_temperature"].units + " / s"},
                ),
                "dQ2": xr.DataArray(
                    dQ2,
                    dims=X["specific_humidity"].dims,
                    coords=X["specific_humidity"].coords,
                    attrs={"units": X["specific_humidity"].units + " / s"},
                ),
            }
        )

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.safe_dump(
                        {
                            "sample_dim_name": self.sample_dim_name,
                            "input_variables": self.input_variables,
                            "output_variables": self.output_variables,
                        }
                    )
                )

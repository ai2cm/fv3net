from typing import Dict, Any, Iterable, Hashable, Optional, Sequence, Tuple
import tensorflow as tf
import xarray as xr
import os
from ..._shared import Predictor, ArrayPacker, io
from ._filesystem import get_dir, put_dir
from .normalizer import LayerStandardScaler
import yaml
import numpy as np


class BPTTModel(Predictor):

    TIME_DIM_NAME = "time"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        n_units: int,
        n_hidden_layers: int,
        kernel_regularizer,
        train_batch_size: int = 512,
        optimizer="adam",
        use_moisture_limiter: bool = False,
        state_noise: float = 0.0,
    ):
        """
        Prognostic variables are air_temperature and specific_humidity.

        Predicted outputs are dQ1 and dQ2, in units per second.

        Args:
            sample_dim_name: name to use for sample dimension
            input_variables: non-prognostic variables to use as inputs
            n_units: number of neurons in each hidden layer
            n_hidden_layers: number of hidden layers
            kernel_regularizer: if given, use for regularizing
                hidden layer kernels
            train_batch_size: batch size to use for keras model training
            optimizer: optimizer for keras training
            use_moisture_limiter: if True, prevent the moisture tendency from
                depleting all moisture on its own in one training timestep.
                Moisture can still go negative if there is a negative
                physics tendency of moisture.
            state_noise: amount of Gaussian noise to add to model prognostic state
                before using it to predict tendencies
        """
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
        self.tendency_scaler = LayerStandardScaler()
        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        # self.tendency_ratio = tendency_ratio
        self.kernel_regularizer = kernel_regularizer
        self.train_batch_size = train_batch_size
        self.optimizer = optimizer
        self.activation = "relu"
        self.train_keras_model: Optional[tf.keras.Model] = None
        self.train_tendency_model: Optional[tf.keras.Model] = None
        self.predict_keras_model: Optional[tf.keras.Model] = None
        self.use_moisture_limiter = use_moisture_limiter
        self.state_noise = state_noise

    def build_for(self, X: xr.Dataset):
        """
        Given a dataset with [sample, time, z] and [sample, time] arrays
        containing data in sample windows, fit the scalers and packers
        and define the keras model (e.g. with the given window length).
        Timestep is determined from the difference between the first two
        times in the dataset.
        """
        if self.train_keras_model is not None:
            raise RuntimeError("cannot build, model is already built!")
        inputs = self.input_packer.to_array(X, is_3d=True)
        state = self.prognostic_packer.to_array(X, is_3d=True)
        self.input_scaler.fit(inputs)
        self.prognostic_scaler.fit(state)
        self.tendency_scaler.std = self.prognostic_scaler.std
        self.tendency_scaler.mean = np.zeros_like(self.prognostic_scaler.mean)
        time = X[BPTTModel.TIME_DIM_NAME]
        self._train_timestep_seconds = (
            time[1].values.item() - time[0].values.item()
        ).total_seconds()
        (
            self.train_keras_model,
            self.train_tendency_model,
            self.predict_keras_model,
        ) = self.build(len(time) - 1)

    @property
    def losses(self):
        assert tuple(self.prognostic_packer.pack_names) == (
            "air_temperature",
            "specific_humidity",
        )
        std = self.prognostic_packer.to_dataset(self.prognostic_scaler.std[None, :])

        air_temperature_factor = tf.constant(
            0.5 / np.mean(std["air_temperature"].values ** 2), dtype=tf.float32
        )
        specific_humidity_factor = tf.constant(
            0.5 / np.mean(std["specific_humidity"].values ** 2), dtype=tf.float32
        )

        def air_temperature_loss(y_true, y_pred):
            return tf.math.multiply(
                air_temperature_factor, tf.losses.mse(y_true, y_pred)
            )

        def specific_humidity_loss(y_true, y_pred):
            return tf.math.multiply(
                specific_humidity_factor, tf.losses.mse(y_true, y_pred)
            )

        return air_temperature_loss, specific_humidity_loss

    def build(
        self, n_window: int
    ) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """
        Define models to use for training and prediction.

        Only train_keras_model is compiled for training.

        Args:
            n_window: number of timesteps to use for training

        Returns:
            train_keras_model: model to use for training
            train_tendencies_keras_model: model to diagnose tendencies of training model
            predict_model: model which predicts tendencies at a specific point in time,
                and shares weights with the training model
        """

        def get_vector(packer, scaler, series=True):
            features = [packer.feature_counts[name] for name in packer.pack_names]
            if series:
                layer_shape=[n_window, n_features]
            else:
                layer_shape=[n_features]
            input_layers = [
                tf.keras.layers.Input(shape=layer_shape) for n_features in features
            ]
            packed = packer.pack_layer()(input_layers)
            if scaler is not None:
                packed = scaler.normalize_layer(packed)
            return input_layers, packed

        input_series_layers, forcing_series_input = get_vector(
            self.input_packer, self.input_scaler, series=True
        )
        state_layers, state_input = get_vector(
            self.prognostic_packer, None, series=False
        )
        given_tendency_series_layers, given_tendency_series_input = get_vector(
            self.prognostic_packer, None, series=True
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
        output_layer = tf.keras.layers.Dense(
            state_input.shape[-1], name="tendency_output"
        )
        train_timestep = np.asarray(self._train_timestep_seconds, dtype=np.float64)
        timestep_divide_layer = tf.keras.layers.Lambda(
            lambda x: x / train_timestep, name="timestep_divide",
        )

        def moisture_tendency_limiter(tendency, state):
            unpack = self.prognostic_packer.unpack_layer(feature_dim=1)
            pack = self.prognostic_packer.pack_layer()
            dQ1, dQ2 = unpack(tendency)
            _, Q2 = unpack(state)

            def limit(args):
                delta_q, q = args
                min_delta_q = tf.where(q > 0, -q, 0.0)
                return tf.where(delta_q > min_delta_q, delta_q, min_delta_q)

            dQ2 = tf.keras.layers.Lambda(limit)([dQ2, Q2])
            return pack([dQ1, dQ2])

        def get_predicted_tendency(x, state):
            for layer in dense_layers:
                x = layer(x)
            x = output_layer(x)
            x = self.tendency_scaler.denormalize_layer(x)
            if self.use_moisture_limiter:
                x = moisture_tendency_limiter(x, state)
            x = timestep_divide_layer(x)
            return x

        tendency_add_layer = tf.keras.layers.Add(name="tendency_add")
        state_add_layer = tf.keras.layers.Add(name="state_add")
        timestep_multiply_layer = tf.keras.layers.Lambda(
            lambda x: x * train_timestep, name="timestep_multiply",
        )
        add_time_dim_layer = tf.keras.layers.Lambda(lambda x: x[:, None, :])

        state_outputs_list = []
        tendency_outputs_list = []
        state = state_input
        for i in range(n_window):
            norm_state = self.prognostic_scaler.normalize(state)
            if self.state_noise > 0.0:
                norm_state = tf.keras.layers.GaussianNoise(self.state_noise)(norm_state)
            x = prepend_forcings(norm_state, i)
            predicted_tendency = get_predicted_tendency(x, state)
            tendency_outputs_list.append(add_time_dim_layer(predicted_tendency))
            given_tendency = get_given_tendency(i)
            total_tendency = tendency_add_layer([predicted_tendency, given_tendency])
            state_update = timestep_multiply_layer(total_tendency)
            state = state_add_layer([state, state_update])
            state_outputs_list.append(add_time_dim_layer(state))

        state_output = tf.keras.layers.concatenate(state_outputs_list, axis=1)
        state_outputs = self.prognostic_packer.unpack_layer(feature_dim=2)(state_output)
        tendency_output = tf.keras.layers.concatenate(tendency_outputs_list, axis=1)
        tendency_outputs = self.prognostic_packer.unpack_layer(feature_dim=2)(
            tendency_output
        )

        train_keras_model = tf.keras.Model(
            inputs=input_series_layers + state_layers + given_tendency_series_layers,
            outputs=state_outputs,
        )
        train_tendency_keras_model = tf.keras.Model(
            inputs=input_series_layers + state_layers + given_tendency_series_layers,
            outputs=tendency_outputs,
        )

        input_layers, forcing_input = get_vector(
            self.input_packer, self.input_scaler, series=False
        )
        state_layers, state_input = get_vector(
            self.prognostic_packer, self.prognostic_scaler, series=False
        )
        denormalized_state = self.prognostic_packer.pack_layer()(state_layers)
        x = tf.keras.layers.concatenate([forcing_input, state_input])
        predicted_tendency = get_predicted_tendency(x, denormalized_state)

        tendency_outputs = self.prognostic_packer.unpack_layer(feature_dim=1)(
            predicted_tendency
        )

        predict_keras_model = tf.keras.Model(
            inputs=input_layers + state_layers, outputs=tendency_outputs
        )

        train_keras_model.compile(optimizer=self.optimizer, loss=self.losses)

        return train_keras_model, train_tendency_keras_model, predict_keras_model

    def fit(self, X: xr.Dataset, *, epochs=1):
        """
        Args:
            X: training data
            epochs: how many epochs to train. default is 1 for the case when X
                contains one batch at a time
        """
        if self.train_keras_model is None:
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

        self.train_keras_model.fit(
            x=self.get_keras_inputs(X),
            y=self.get_target_state(X),
            batch_size=self.train_batch_size,
            epochs=epochs,
            shuffle=True,
        )

    def loss(self, X: xr.Dataset):
        """
        Return the loss on a sequence dataset.

        Useful for getting validation loss.
        """
        if self.train_keras_model is None:
            raise RuntimeError("must build model before loss can be calculated")
        else:
            output = self.train_keras_model.predict(self.get_keras_inputs(X))
            target = self.get_target_state(X)
            loss = sum(
                np.mean(f(truth, pred))
                for f, truth, pred in zip(self.losses, target, output)
            )
            return loss

    def get_keras_inputs(self, X) -> Sequence[np.ndarray]:
        """
        Return numpy arrays to be passed in to the Keras training model.
        """

        def get_inputs(X):
            return get_keras_arrays(X, self.input_packer.pack_names, slice(0, -1))

        def get_initial_state(X):
            return get_keras_arrays(X, self.prognostic_packer.pack_names, 0)

        def get_given_tendency(X):
            return get_keras_arrays(X, self.given_tendency_names, slice(0, -1))

        return get_inputs(X) + get_initial_state(X) + get_given_tendency(X)

    def get_target_state(self, X):
        return get_keras_arrays(X, self.prognostic_packer.pack_names, slice(1, None))

    @classmethod
    def load(cls, path):
        raise NotImplementedError(
            "class meant for training only, "
            "should be using self.predictor_model to load/predict"
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "class meant for training only, "
            "should be using self.predictor_model to load/predict"
        )

    @property
    def predictor_model(self) -> "StepwiseModel":
        predictor_model = StepwiseModel(
            self.sample_dim_name,
            tuple(self.input_packer.pack_names)
            + tuple(self.prognostic_packer.pack_names),
            self.output_variables,
            self.predict_keras_model,
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


@io.register("stepwise-keras")
class StepwiseModel(AllKerasModel):
    def integrate_stepwise(self, ds):
        time = ds["time"]
        timestep_seconds = (
            time[1].values.item() - time[0].values.item()
        ).total_seconds()

        state = {
            "air_temperature": ds["air_temperature"].isel(time=0).values,
            "specific_humidity": ds["specific_humidity"].isel(time=0).values,
        }
        state_out_list = []
        n_timesteps = len(ds["time"]) - 1
        for i in range(n_timesteps):
            print(f"Step {i+1} of {n_timesteps}")
            input_ds = ds.isel(time=i)
            input_ds["air_temperature"] = xr.DataArray(
                state["air_temperature"],
                dims=[self.sample_dim_name, "z"],
                attrs={"units": ds["air_temperature"].units},
            )
            input_ds["specific_humidity"] = xr.DataArray(
                state["specific_humidity"],
                dims=[self.sample_dim_name, "z"],
                attrs={"units": ds["specific_humidity"].units},
            )
            tendency_ds = self.predict(input_ds)
            state["air_temperature"] = (
                state["air_temperature"]
                + (
                    tendency_ds["dQ1"].values
                    + input_ds["air_temperature_tendency_due_to_model"].values
                )
                * timestep_seconds
            )
            state["specific_humidity"] = (
                state["specific_humidity"]
                + (
                    tendency_ds["dQ2"]
                    + input_ds["specific_humidity_tendency_due_to_model"].values
                )
                * timestep_seconds
            )
            data_vars = {}
            for name, value in state.items():
                data_vars[name] = ([self.sample_dim_name, "z"], value)
            for name in (
                "air_temperature_tendency_due_to_model",
                "specific_humidity_tendency_due_to_model",
                "air_temperature_tendency_due_to_nudging",
                "specific_humidity_tendency_due_to_nudging",
            ):
                data_vars[name] = input_ds[name]
            for name in state.keys():
                data_vars[f"{name}_reference"] = (
                    ds[name].isel(time=i + 1).reset_coords(drop=True)
                )
            for name in ("dQ1", "dQ2"):
                data_vars[name] = tendency_ds[name]
            timestep_ds = xr.Dataset(data_vars=data_vars)
            state_out_list.append(timestep_ds)
        return xr.concat(state_out_list, dim="time").transpose(
            self.sample_dim_name, "time", "z"
        )

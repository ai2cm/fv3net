from typing import Iterable, Optional, Mapping, Union
import functools
import yaml
import os
import logging
import fv3gfs.wrapper
import fv3gfs.util
import tensorflow as tf
import tensorflow.keras.backend as K
import fv3fit._shared.packer
import fv3fit.keras._models.models
import fv3fit
import numpy as np
import vcm.safe
import xarray as xr
from datetime import timedelta
import horovod.tensorflow.keras as hvd
from mpi4py import MPI

MODEL_OUTPUT_PATH = "model"


logger = logging.getLogger(__name__)


class GCMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units: int, n_output: int, n_hidden_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.activation = tf.keras.activations.relu

    @property
    def state_size(self):
        return self.n_output

    @property
    def output_size(self):
        return self.n_output

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "expected to be used in stateful one-batch-at-a-time mode, "
                f"got input_shape {input_shape}"
            )
        n_input = input_shape[1] + self.n_output  # prognostic state also used as input
        kernel = self.add_weight(
            shape=(n_input, self.units),
            initializer="glorot_uniform",
            name=f"dense_kernel_0",
        )
        bias = self.add_weight(
            shape=(self.units,), initializer="zeros", name="dense_bias_0"
        )
        self.dense_weights = [(kernel, bias)]
        for i in range(1, self.n_hidden_layers):
            kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer="glorot_uniform",
                name=f"dense_kernel_{i}",
            )
            bias = self.add_weight(
                shape=(self.units,), initializer="zeros", name=f"dense_bias_{i}"
            )
            self.dense_weights.append((kernel, bias))
        self.output_kernel = self.add_weight(
            shape=(self.units, self.n_output),
            initializer="glorot_uniform",
            name=f"output_kernel",
        )
        self.output_bias = self.add_weight(
            shape=(self.n_output,), initializer="zeros", name="output_bias"
        )
        self.built = True

    def call(self, inputs, states):
        gcm_state = states[0]
        h = K.concatenate([inputs, gcm_state])
        for kernel, bias in self.dense_weights:
            h = self.activation(K.dot(h, kernel) + bias)
        tendency_output = K.dot(h, self.output_kernel) + self.output_bias
        # tendencies are on a much smaller scale than the variability of the data
        gcm_update = tf.math.multiply(
            tf.constant(0.1, dtype=tf.float32), tendency_output
        )
        gcm_output = gcm_state + gcm_update
        return gcm_output, [gcm_output]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_output": self.n_output,
                "n_hidden_layers": self.n_hidden_layers,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RecurrentMPIModel(fv3fit.keras._models.models.PackedKerasModel):
    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
    ):
        super().__init__(
            sample_dim_name,
            input_variables,
            output_variables,
            weights=weights,
            normalize_loss=normalize_loss,
        )

    def fit_array(self, X):
        callbacks = [
            # Horovod: broadcast initial variable states from rank 0
            # to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        ]

        # Horovod: save checkpoints only on worker 0 to prevent
        # other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint("./checkpoint-{epoch}.h5")
            )
        self.model.fit(X, callbacks=callbacks)


class RecurrentGCMModel(RecurrentMPIModel):

    _SAMPLE_DIM_NAME = "sample"

    def __init__(
        self,
        comm,
        input_variables: Iterable[str],
        prognostic_variables: Iterable[str],
        units=64,
        n_hidden_layers=3,
        optimizer=None,
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        negative_water_weight: float = 1.0,
    ):
        # X is input, y is prognostic
        self.n_batch = None
        self.comm = comm
        self.units = units
        self.n_hidden_layers = n_hidden_layers
        self._model = None
        self._rnn = None
        self._last_state = None
        self.negative_water_weight = negative_water_weight
        if optimizer is None:
            self.optimizer = hvd.DistributedOptimizer(
                tf.keras.optimizers.Adam(0.002 * hvd.size(), amsgrad=True, clipnorm=1.0)
            )
        else:
            self.optimizer = hvd.DistributedOptimizer(optimizer)
        super().__init__(
            self._SAMPLE_DIM_NAME,
            input_variables,
            prognostic_variables,
            weights=weights,
            normalize_loss=normalize_loss,
        )

    @property
    def rnn(self):
        return self.model.get_layer("rnn")

    def adapt(self, state):
        """prepare model dimensions and normalization according to an example state"""
        if self._model is not None:
            raise NotImplementedError("cannot adapt twice")
        all_variables = list(self.input_variables) + (self.output_variables)
        minimal_state = {name: state[name] for name in all_variables}
        ds = _stack_samples(
            fv3gfs.util.to_dataset(minimal_state), self._SAMPLE_DIM_NAME
        )
        X_in = self.X_packer.to_array(ds)
        y_initial = self.y_packer.to_array(ds)
        n_features_in, n_features_out = X_in.shape[-1], y_initial.shape[-1]
        self._fit_normalization(X_in, y_initial)
        self.n_batch = X_in.shape[0]
        self._model = self.get_model(n_features_in, n_features_out)

    def _fit_normalization(self, X, y):
        super()._fit_normalization(X, y)
        self.comm.Allreduce(
            self.X_scaler.mean / self.comm.Get_size(), self.X_scaler.mean, op=MPI.SUM
        )
        self.comm.Allreduce(
            self.X_scaler.std / self.comm.Get_size(), self.X_scaler.std, op=MPI.SUM
        )
        self.comm.Allreduce(
            self.y_scaler.mean / self.comm.Get_size(), self.y_scaler.mean, op=MPI.SUM
        )
        self.comm.Allreduce(
            self.y_scaler.std / self.comm.Get_size(), self.y_scaler.std, op=MPI.SUM
        )

    def get_model(self, n_features_in, n_features_out):
        n_input = n_features_in
        n_state = n_features_out
        rnn = tf.keras.layers.RNN(
            [
                GCMCell(
                    units=self.units,
                    n_output=n_state,
                    n_hidden_layers=self.n_hidden_layers,
                )
            ],
            stateful=True,
            batch_input_shape=[self.n_batch, 1, n_input],
            name="rnn",
        )
        inputs = tf.keras.layers.Input(batch_shape=[self.n_batch, n_input])
        x = self.X_scaler.normalize_layer(inputs)
        x = tf.keras.layers.Reshape([1, n_input], input_shape=[n_input])(
            x
        )  # [timestep, channels]
        x = rnn(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = self.y_scaler.denormalize_layer(x)
        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            experimental_run_tf_function=False,
        )
        return model

    def fit(self, *args, **kwargs):
        raise NotImplementedError("use fit_state instead")

    def fit_state(self, state, target_state):
        if self._model is None:
            raise RuntimeError("call model.adapt before model.fit_state")
        all_variables = list(self.input_variables) + (self.output_variables)
        minimal_state = {name: state[name] for name in all_variables}
        minimal_target_state = {
            name: target_state[name] for name in self.output_variables
        }
        ds = _stack_samples(
            fv3gfs.util.to_dataset(minimal_state), self._SAMPLE_DIM_NAME
        )
        target_ds = _stack_samples(
            fv3gfs.util.to_dataset(minimal_target_state), self._SAMPLE_DIM_NAME
        )
        X_in = self.X_packer.to_array(ds)
        y_model = self.y_packer.to_array(ds)
        y_model_norm = self.y_scaler.normalize(y_model)
        if self._last_state is not None:
            # add dynamics and physics tendencies
            base_model_diff = tf.constant(
                y_model_norm - self._last_state, dtype=tf.float32
            )
            self.rnn.states[0].assign_add(base_model_diff)
        self._last_state = y_model_norm
        y_target = self.y_packer.to_array(target_ds)
        self.model.fit(
            X_in, y_target, epochs=1, batch_size=X_in.shape[0], shuffle=False
        )
        coarse_loss = self.comm.reduce(
            self.loss(y_target, y_model) / self.comm.Get_size(), MPI.SUM
        )
        if self.comm.Get_rank() == 0:
            print(f"coarse model loss: {coarse_loss}")

    @property
    def loss(self):
        def custom_loss(y_true, y_pred):
            # we can assume temperature will never be even close to zero
            # TODO this assumes temperature + humidity are the prognostic outputs,
            # better would be to get the specific indices corresponding to humidity
            negative_water = tf.math.multiply(
                tf.constant(self.negative_water_weight, dtype=tf.float32),
                tf.math.reduce_mean(
                    tf.where(
                        y_pred < tf.constant(0.0, dtype=tf.float32),
                        tf.math.multiply(tf.constant(-1.0, dtype=tf.float32), y_pred),
                        tf.constant(0.0, dtype=tf.float32),
                    )
                ),
            )
            base_loss = tf.cast(
                super(RecurrentGCMModel, self).loss(y_true, y_pred), tf.float32
            )
            return tf.math.add(negative_water, base_loss)

        return custom_loss

    def get_state(self) -> dict:
        y = self.y_scaler.denormalize(self.rnn.states[0].numpy()).astype(np.float64)
        ds = self.y_packer.to_dataset(y).unstack(self._SAMPLE_DIM_NAME)
        state = {
            name: fv3gfs.util.Quantity.from_data_array(value)
            for name, value in ds.data_vars.items()
        }
        for quantity in state.values():  # remove non-negative humidity/temperature
            quantity.data[quantity.data < 0.0] = 0.0
        return state

    def set_state(self, state):
        """Updates the internal RNN state to a target atmospheric state."""
        self._last_state = None
        minimal_state = {name: state[name] for name in self.output_variables}
        ds = _stack_samples(
            fv3gfs.util.to_dataset(minimal_state), self._SAMPLE_DIM_NAME
        )
        y = self.y_scaler.normalize(self.y_packer.to_array(ds))
        self.rnn.reset_states(states=[tf.constant(y, dtype=tf.float32)])
        # state_diff = tf.constant(y - self.rnn.states[0].numpy(), dtype=tf.float32)
        # self.rnn.states[0].assign_add(state_diff)


def nudge_to_reference(state, reference, timescales, timestep):
    tendencies = fv3gfs.util.apply_nudging(state, reference, timescales, timestep)
    tendencies = append_key_label(tendencies, "_tendency_due_to_nudging")
    tendencies["time"] = state["time"]
    return tendencies


def append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


def _stack_samples(ds: xr.Dataset, sample_dim_name: str) -> xr.Dataset:
    stack_dims = [dim for dim in ds.dims if dim not in fv3gfs.util.Z_DIMS]
    if len(set(ds.dims).intersection(fv3gfs.util.Z_DIMS)) > 1:
        raise ValueError("Data cannot have >1 vertical dimension, has {ds.dims}.")
    ds_stacked = vcm.safe.stack_once(
        ds, sample_dim_name, stack_dims, allowed_broadcast_dims=fv3gfs.util.Z_DIMS,
    )
    return ds_stacked.transpose()


def time_to_label(time):
    return (
        f"{time.year:04d}{time.month:02d}{time.day:02d}."
        f"{time.hour:02d}{time.minute:02d}{time.second:02d}"
    )


def get_restart_directory(reference_dir, label):
    return os.path.join(reference_dir, label)


def get_reference_state(time, reference_dir, communicator, only_names):
    label = time_to_label(time)
    dirname = get_restart_directory(reference_dir, label)
    logger.debug(f"Restart dir: {dirname}")
    state = fv3gfs.wrapper.open_restart(
        dirname, communicator, label=label, only_names=only_names
    )
    state["time"] = time
    return state


fv3fit.keras._models.models.PackedKerasModel.custom_objects["GCMCell"] = GCMCell


def bcast_from_root(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            result = func(*args, **kwargs)
        else:
            result = None
        return MPI.COMM_WORLD.bcast(result)

    return wrapped


def get_timescales_from_config(config):
    return_dict = {}
    for name, hours in config["nudging"]["timescale_hours"].items():
        return_dict[name] = timedelta(seconds=int(hours * 60 * 60))
    return return_dict


@bcast_from_root
def load_config(filename):
    with open("fv3config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    logging.basicConfig(level=logging.INFO)
    config = load_config("fv3config.yml")
    reference_dir = config["nudging"]["restarts_path"]
    partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(config["namelist"])
    communicator = fv3gfs.util.CubedSphereCommunicator(MPI.COMM_WORLD, partitioner)
    nudging_timescales = get_timescales_from_config(config)
    nudging_variables = list(nudging_timescales.keys())
    timestep = get_timestep(config)
    reference_dir = (
        "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs"
        "/restarts/C48"
    )
    nudge = functools.partial(
        nudge_to_reference, timescales=nudging_timescales, timestep=timestep,
    )
    input_variables = [
        "surface_geopotential",
        "surface_pressure",
        "mean_cos_zenith_angle",
        "land_sea_mask",
        "latent_heat_flux",
        "sensible_heat_flux",
        "vertical_wind",
    ]
    prognostic_variables = [
        "air_temperature",
        "specific_humidity",
    ]
    all_variables = list(
        set(input_variables + prognostic_variables + nudging_variables)
    )
    hvd.init(comm=MPI.COMM_WORLD)
    assert hvd.mpi_threads_supported()
    fv3gfs.wrapper.initialize()
    initial_state = fv3gfs.wrapper.get_state(["time"] + all_variables)
    if os.path.exists(MODEL_OUTPUT_PATH):
        model = RecurrentGCMModel.load(MODEL_OUTPUT_PATH)
    else:
        model = RecurrentGCMModel(
            MPI.COMM_WORLD,
            input_variables=input_variables,
            prognostic_variables=prognostic_variables,
            units=128,
        )
        model.adapt(initial_state)
    model.set_state(initial_state)
    start_time = initial_state["time"]
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
        state = fv3gfs.wrapper.get_state(["time"] + all_variables)
        if rank == 0:
            time_elapsed = state["time"] - start_time
            print(f"time elapsed: {time_elapsed}")
        reference = get_reference_state(
            state["time"],
            reference_dir,
            communicator,
            only_names=list(set(prognostic_variables + nudging_variables)),
        )
        if i % (4 * 24 * 5) == 0:  # every 5 days
            if rank == 0:
                print("resetting ML model state")
            model.set_state(state)
        model.fit_state(state, reference)
        # apply nudging
        # if i > 24:  # 6 hours
        #     if rank == 0:
        #         print('copying ML model state to Fortran')
        #     state.update(model.get_state())
        nudge(state, reference)
        updated_state_members = {key: state[key] for key in nudging_variables}
        fv3gfs.wrapper.set_state(updated_state_members)
    if MPI.COMM_WORLD.Get_rank() == 0:
        model.dump(MODEL_OUTPUT_PATH)
    hvd.shutdown()
    fv3gfs.wrapper.cleanup()

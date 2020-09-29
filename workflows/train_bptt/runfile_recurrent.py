from typing import Iterable, Optional, Mapping, Union
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
import horovod.tensorflow.keras as hvd
from mpi4py import MPI
import f90nml

MODEL_OUTPUT_PATH = "model"


logger = logging.getLogger(__name__)


class GCMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units: int, n_output: int, n_hidden_layers: int, **kwargs):
        self.units = units
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.activation = tf.keras.activations.relu
        super().__init__(**kwargs)

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
        gcm_output = gcm_state + tendency_output
        return gcm_output, [gcm_output]

    def get_config(self):
        return {"units": self.units}


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
    ):
        # X is input, y is prognostic
        self.n_batch = None
        self.comm = comm
        self.units = units
        self.n_hidden_layers = n_hidden_layers
        self._model = None
        self.rnn = None
        if optimizer is None:
            self.optimizer = hvd.DistributedOptimizer(
                tf.keras.optimizers.Adam(0.001 / hvd.size())
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
        if self.rnn is None:
            self.rnn = tf.keras.layers.RNN(
                [
                    GCMCell(
                        units=self.units,
                        n_output=n_state,
                        n_hidden_layers=self.n_hidden_layers,
                    )
                ],
                stateful=True,
                batch_input_shape=[self.n_batch, 1, n_input],
            )
        inputs = tf.keras.layers.Input(batch_shape=[self.n_batch, n_input])
        x = self.X_scaler.normalize_layer(inputs)
        x = tf.keras.layers.Reshape([1, n_input], input_shape=[n_input])(
            x
        )  # [timestep, channels]
        x = self.rnn(x)
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
        y_target = self.y_packer.to_array(target_ds)
        self.model.fit(
            X_in, y_target, epochs=1, batch_size=X_in.shape[0], shuffle=False
        )

    def get_state(self) -> dict:
        y = self.y_scaler.denormalize(self.rnn.states[0].numpy()).astype(np.float64)
        ds = self.y_packer.to_dataset(y).unstack(self._SAMPLE_DIM_NAME)
        state = {
            name: fv3gfs.util.Quantity.from_data_array(value)
            for name, value in ds.data_vars.items()
        }
        return state

    def set_state(self, state):
        """Updates the internal RNN state to a target atmospheric state."""
        minimal_state = {name: state[name] for name in self.output_variables}
        ds = _stack_samples(
            fv3gfs.util.to_dataset(minimal_state), self._SAMPLE_DIM_NAME
        )
        y = self.y_scaler.normalize(self.y_packer.to_array(ds))
        state_diff = tf.constant(y - self.rnn.states[0].numpy(), dtype=tf.float32)
        self.rnn.states[0].assign_add(state_diff)


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


if __name__ == "__main__":
    reference_dir = (
        "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs"
        "/restarts/C48"
    )
    namelist = f90nml.read("input.nml")
    partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(namelist)
    communicator = fv3gfs.util.CubedSphereCommunicator(MPI.COMM_WORLD, partitioner)
    input_variables = [
        "surface_geopotential",
        "surface_pressure",
        "mean_cos_zenith_angle",
        "land_sea_mask",
    ]
    prognostic_variables = [
        "air_temperature",
        "specific_humidity",
    ]
    all_variables = input_variables + prognostic_variables
    hvd.init(comm=MPI.COMM_WORLD)
    assert hvd.mpi_threads_supported()
    fv3gfs.wrapper.initialize()
    initial_state = fv3gfs.wrapper.get_state(all_variables)
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
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
        state = fv3gfs.wrapper.get_state(["time"] + all_variables)
        reference = get_reference_state(
            state["time"], reference_dir, communicator, only_names=prognostic_variables
        )
        model.set_state(state)
        model.fit_state(state, reference)
        # fv3gfs.wrapper.set_state(model.get_state())
    if MPI.COMM_WORLD.Get_rank() == 0:
        model.dump(MODEL_OUTPUT_PATH)
    hvd.shutdown()
    fv3gfs.wrapper.cleanup()

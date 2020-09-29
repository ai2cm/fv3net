from typing import Iterable, Optional, Mapping, Union
import fv3gfs.wrapper
import fv3gfs.util
import tensorflow as tf
import fv3fit.keras._models.models
import numpy as np
import vcm.safe
import xarray as xr
import horovod.tensorflow.keras as hvd
from mpi4py import MPI

MODEL_OUTPUT_PATH = "model"


class DenseMPIModel(fv3fit.keras._models.models.PackedKerasModel):
    def __init__(
        self,
        comm,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        hidden: int = 3,
        width: int = 16,
    ):
        self._comm = comm
        self._hidden = hidden
        self._width = width
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

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for _ in range(self._hidden):
            x = tf.keras.layers.Dense(
                self._width, activation=tf.keras.activations.relu
            )(x)
        x = tf.keras.layers.Dense(n_features_out, activation=tf.keras.activations.relu)(
            x
        )
        outputs = self.y_scaler.denormalize_layer(x)
        model = tf.keras.Model(self._comm, inputs=inputs, outputs=outputs)
        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(
            optimizer=hvd.DistributedOptimizer(
                tf.keras.optimizers.Adam(0.001 / hvd.size())
            ),
            loss=self.loss,
            experimental_run_tf_function=False,
        )
        return model


def _stack_samples(ds: xr.Dataset, sample_dim_name: str) -> xr.Dataset:
    stack_dims = [dim for dim in ds.dims if dim not in fv3gfs.util.Z_DIMS]
    if len(set(ds.dims).intersection(fv3gfs.util.Z_DIMS)) > 1:
        raise ValueError("Data cannot have >1 vertical dimension, has {ds.dims}.")
    ds_stacked = vcm.safe.stack_once(
        ds, sample_dim_name, stack_dims, allowed_broadcast_dims=fv3gfs.util.Z_DIMS,
    )
    return ds_stacked.transpose()


class OnlineModel:

    _SAMPLE_DIM_NAME = "sample"

    def __init__(
        self,
        comm,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        **kwargs,
    ):
        self.xr = DenseMPIModel(
            comm,
            self._SAMPLE_DIM_NAME,
            input_variables,
            output_variables,
            weights=weights,
            normalize_loss=normalize_loss,
            **kwargs,
        )

    def fit(self, state) -> None:
        ds = _stack_samples(fv3gfs.util.to_dataset(state), self._SAMPLE_DIM_NAME)
        self.xr.fit([ds])

    def dump(self, path: str) -> None:
        self.xr.dump(path)

    @classmethod
    def load(cls, path: str) -> "OnlineModel":
        obj = cls([], [])
        obj.xr = DenseMPIModel.load(path)


if __name__ == "__main__":
    hvd.init(comm=MPI.COMM_WORLD)
    assert hvd.mpi_threads_supported()
    model = OnlineModel(
        MPI.COMM_WORLD,
        input_variables=["air_temperature"],
        output_variables=["specific_humidity"],
    )
    fv3gfs.wrapper.initialize()
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
        state = fv3gfs.wrapper.get_state(["air_temperature", "specific_humidity"])
        model.fit(state)
    if MPI.COMM_WORLD.Get_rank() == 0:
        model.dump(MODEL_OUTPUT_PATH)
    hvd.shutdown()
    fv3gfs.wrapper.cleanup()

from typing import Iterable, Hashable, Sequence, Tuple
import fv3gfs.util
import numpy as np
import fv3fit
import xarray as xr
import tensorflow as tf
import argparse
import concurrent.futures
from download import INPUT_NAMES
import copy
import random
import vcm


def penalize_negative_water(loss, negative_water_weight, negative_water_threshold):
    """
    negative_water_threshold should have dimension [nz].

    Assumes water is the last variable in the output prediction, uses
    the shape of the negative water threshold to determine number of water features.
    """
    nz = negative_water_threshold.shape[0]
    negative_water_threshold = tf.constant(negative_water_threshold, dtype=tf.float32)

    def custom_loss(y_true, y_pred):
        # we can assume temperature will never be even close to zero
        # TODO this assumes temperature + humidity are the prognostic outputs,
        # better would be to get the specific indices corresponding to humidity
        if len(y_pred.shape) == 2:
            pred_water = y_pred[:, -nz:]
            shaped_threshold = negative_water_threshold[None, :]
        elif len(y_pred.shape) == 3:
            pred_water = y_pred[:, :, -nz:]
            shaped_threshold = negative_water_threshold[None, None, :]
        else:
            raise NotImplementedError("only 2d or 3d y are supported")
        negative_water = tf.math.multiply(
            tf.constant(negative_water_weight, dtype=tf.float32),
            tf.math.reduce_mean(
                tf.where(
                    pred_water < shaped_threshold,
                    tf.math.multiply(
                        tf.constant(-1.0, dtype=tf.float32),
                        pred_water - shaped_threshold,
                    ),
                    tf.constant(0.0, dtype=tf.float32),
                )
            ),
        )
        base_loss = tf.cast(loss(y_true, y_pred), tf.float32)
        return tf.math.add(negative_water, base_loss)

    return custom_loss


class PreloadedIterator:
    """
    Iterator for data which asynchronously pre-loads
    the next set of output.
    """

    def __init__(
        self, filenames: Sequence[str], loader_function,
    ):
        """
        Args:
            filenames: npz files containing TrainingArrays data
        """
        self.loader_function = loader_function
        self.filenames = filenames
        # this will be shuffled at the start of each iteration
        self._shuffled_filenames = list(filenames)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._idx = 0
        self._load_thread = None
        self._start_load()

    def _start_load(self):
        if self._idx < len(self.filenames):
            self._load_thread = self._executor.submit(
                self.loader_function, self.filenames[self._idx],
            )

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration()
        else:
            arrays = self._load_thread.result()
            self._load_thread = None
            self._idx += 1
            if self._idx < len(self):
                self._start_load()
            return arrays

    def __iter__(self):
        self._idx = 0
        # new shuffled order each time we iterate
        random.shuffle(self._shuffled_filenames)
        if self._load_thread is None:
            self._start_load()
        return self

    def __len__(self):
        return len(self.filenames)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "model_output_dir", type=str, help="directory to write the trained model"
    )
    return parser


def integrate_stepwise(ds, model):
    time = ds["time"]
    timestep_seconds = (
        time[1].values.item() - time[0].values.item()
    ).total_seconds()
    ds = ds.rename({
        "air_temperature_tendency_due_to_nudging": "nQ1",
        "specific_humidity_tendency_due_to_nudging": "nQ2",
        "air_temperature_tendency_due_to_model": "pQ1",
        "specific_humidity_tendency_due_to_model": "pQ2",
    })

    state = {
        "air_temperature": ds["air_temperature"].isel(time=0),
        "specific_humidity": ds["specific_humidity"].isel(time=0),
    }
    state_out_list = []
    n_timesteps = len(ds["time"]) - 1
    for i in range(n_timesteps):
        print(f"Step {i+1} of {n_timesteps}")
        forcing_ds = ds.isel(time=i)
        ref_data = {
            "nQ1": forcing_ds["nQ1"],
            "nQ2": forcing_ds["nQ2"],
            "air_temperature_reference": ds["air_temperature"].isel(time=i+1).reset_coords(drop=True),
            "specific_humidity_reference": ds["specific_humidity"].isel(time=i+1).reset_coords(drop=True),
        }
        forcing_ds.assign(**state)
        tendency_ds = model.predict(forcing_ds)
        assert not np.any(np.isnan(state["air_temperature"].values))
        state["air_temperature"] = state["air_temperature"] + (
            tendency_ds["dQ1"] + forcing_ds["pQ1"]
        ) * timestep_seconds
        state["specific_humidity"] = state["specific_humidity"] + (
            tendency_ds["dQ2"] + forcing_ds["pQ2"]
        ) * timestep_seconds
        tendency_data = {
            "dQ1": tendency_ds["dQ1"],
            "dQ2": tendency_ds["dQ2"],
        }
        timestep_ds = xr.Dataset(data_vars={**state, **ref_data, **tendency_data})
        state_out_list.append(timestep_ds)
    return xr.concat(state_out_list, dim="time")


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    # tf.config.experimental.enable_mlir_graph_optimization()
    parser = get_parser()
    args = parser.parse_args()

    fs = vcm.get_fs(args.arrays_dir)
    filenames = sorted(fs.listdir(args.arrays_dir, detail=False))
    first_filename = filenames[0]

    with open(first_filename, "rb") as f:
        ds = xr.open_dataset(f)
        sample_dim_name = ds["air_temperature"].dims[0]
        model = fv3fit.keras.BPTTModel(
            sample_dim_name,
            INPUT_NAMES,
            n_units=32,
            n_hidden_layers=3,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            train_batch_size=48,
            optimizer=tf.keras.optimizers.Adam(lr=0.001, amsgrad=True, clipnorm=1.0),
        )
        model.build_for(ds)

    training_datasets = PreloadedIterator(
        filenames, loader_function=xr.open_dataset
    )

    base_epoch = 0
    for i_epoch in range(50):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        for i, ds in enumerate(training_datasets):
            model.fit(ds, epochs=1)

    fv3fit.dump(model.predictor_model, args.model_output_dir)

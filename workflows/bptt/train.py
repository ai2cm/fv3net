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


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    # tf.config.experimental.enable_mlir_graph_optimization()
    parser = get_parser()
    args = parser.parse_args()

    fs = vcm.get_fs(args.arrays_dir)
    filenames = fs.listdir(args.arrays_dir, detail=False)
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

    # use last file for validation
    training_datasets = PreloadedIterator(
        filenames[:-1], loader_function=xr.open_dataset
    )

    # with open(filenames[-1], "rb") as f:
    #     validation_ds = xr.open_dataset(f)
    #     validation_ds.load()

    base_epoch = 0
    for i_epoch in range(1):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        for i, ds in enumerate(training_datasets):
            model.fit(ds, epochs=1)

    fv3fit.dump(model.predictor_model, args.model_output_dir)

    # do some basic sanity checks of saved model

    loaded = fv3fit.load(args.model_output_dir)
    with open(first_filename, "rb") as f:
        ds = xr.open_dataset(f).isel(time=0)
        reference_output = model.predictor_model.predict(ds)
        # should not get same prediction for q and T
        assert not np.all(reference_output["dQ1"].values == reference_output["dQ2"].values)
        # test that loaded model gives the same predictions
        loaded_output = loaded.predict(ds)
        for name in reference_output.data_vars.keys():
            xr.testing.assert_equal(reference_output[name], loaded_output[name])

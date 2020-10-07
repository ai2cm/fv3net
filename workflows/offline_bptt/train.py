import os
import fv3fit
import fv3fit.keras._models.loss
import fv3fit.keras._models.normalizer
import loaders
import numpy as np
import tensorflow as tf
import random
import vcm
import fv3gfs.util
import argparse

COARSE_DATA_DIR = (
    "gs://vcm-ml-experiments/2020-06-17-triad-round-1/nudging/nudging/outdir-3h"
)
REF_DATA_DIR = "gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts"
SAMPLE_DIM_NAME = "sample"
TIME_NAME = "time"


def stack(ds):
    stack_dims = [dim for dim in ds.dims if dim not in fv3gfs.util.Z_DIMS]
    if len(set(ds.dims).intersection(fv3gfs.util.Z_DIMS)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = vcm.safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=list(fv3gfs.util.Z_DIMS) + [TIME_NAME],
    )
    return ds_stacked.transpose()


def load_data(
    nudged_mapper,
    ref_mapper,
    X_packer,
    y_packer,
    n_timesteps=None,
    n_timestep_frequency=None,
    label="default",
    cache_dir=None,
):
    use_cache = cache_dir is not None
    shared_keys = sorted(list(set(nudged_mapper.keys()).union(ref_mapper.keys())))
    if n_timestep_frequency is not None:
        shared_keys = shared_keys[::n_timestep_frequency]
    if n_timesteps is not None:
        shared_keys = shared_keys[:n_timesteps]
    X_coarse_list = []
    y_coarse_list = []
    y_ref_list = []

    def pack(packer, ds):
        drop_names = set(ds.data_vars).union(ds.coords).difference(packer.pack_names)
        ds = ds.drop_vars(drop_names)
        return_value = packer.to_array(stack(ds))[:, None, :].astype(np.float32)
        return return_value

    if use_cache:
        data_cache_dir = os.path.join(cache_dir, f"cache-load-{label}")
        if not os.path.exists(data_cache_dir):
            os.mkdir(data_cache_dir)

    for i, timestamp in enumerate(shared_keys):
        print(f"loading timestep {i + 1} of {len(shared_keys)}")
        filename = os.path.join(data_cache_dir, f"timestep_{timestamp}.npz")
        if use_cache and os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = np.load(f)
                X_coarse, y_coarse, y_ref = (data[k] for k in data)
        else:
            ref_ds = ref_mapper[timestamp]
            y_ref = pack(y_packer, ref_ds)
            nudged_ds = nudged_mapper[timestamp]
            X_coarse = pack(X_packer, nudged_ds)
            y_coarse = pack(y_packer, nudged_ds)
            if use_cache:
                with open(filename, "wb") as f:
                    np.savez(f, X_coarse, y_coarse, y_ref)
        X_coarse_list.append(X_coarse)
        y_coarse_list.append(y_coarse)
        y_ref_list.append(y_ref)
    X_coarse = np.concatenate(X_coarse_list, axis=1)
    del X_coarse_list  # free up duplicated memory after concatenating
    y_coarse = np.concatenate(y_coarse_list, axis=1)
    del y_coarse_list
    y_ref = np.concatenate(y_ref_list, axis=1)
    return X_coarse, y_coarse, y_ref


class WindowedDataSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_coarse,
        y_coarse,
        y_ref,
        n_window: int,
        n_between_windows: int,
        batch_size: int,
    ):
        if X_coarse.shape[0] % batch_size != 0:
            raise ValueError(
                "samples must be evenly divisible by batch_size, "
                f"got shape {X_coarse.shape} and batch_size {batch_size}"
            )
        self.X_coarse = X_coarse
        self.y_coarse = y_coarse
        self.y_ref = y_ref
        self.n_window = n_window
        self.n_between_windows = n_between_windows
        self._n_batches = X_coarse.shape[0] // batch_size
        self._n_windows = int((X_coarse.shape[1] - n_window) / n_between_windows)
        all_indices = []
        for i_window in range(self._n_windows):
            for i_batch in range(self._n_batches):
                all_indices.append((i_window, i_batch))
        random.shuffle(all_indices)
        self._indices = all_indices
        self.batch_size = batch_size

    def __getitem__(self, key):
        i_window, i_batch = self._indices[key]
        batch_start = i_batch * self.batch_size
        window_start = i_window * self.n_window
        s = (
            slice(batch_start, batch_start + self.batch_size),
            slice(window_start, window_start + self.n_window),
            slice(None, None),
        )
        return ([self.X_coarse[s], self.y_coarse[s]], self.y_ref[s])

    def __len__(self):
        return len(self._indices)


def build_model(
    n_batch, n_input, n_state, n_window, units, n_hidden_layers, X_scaler, y_scaler,
):
    forcing_input = tf.keras.layers.Input(batch_shape=[n_batch, n_window, n_input])
    state_input = tf.keras.layers.Input(batch_shape=[n_batch, n_window, n_state])
    x_forcing = X_scaler.normalize_layer(forcing_input)
    x_state = y_scaler.normalize_layer(state_input)
    rnn = tf.keras.layers.RNN(
        fv3fit.keras.GCMCell(
            units=units,
            n_input=n_input,
            n_state=n_state,
            n_hidden_layers=n_hidden_layers,
        ),
        name="rnn",
        return_sequences=True,
    )
    # x = tf.keras.layers.Concatenate()([x_forcing, x_state])
    x = rnn(inputs=(x_forcing, x_state))
    outputs = y_scaler.denormalize_layer(x)
    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model = tf.keras.Model(inputs=[forcing_input, state_input], outputs=outputs)
    return model


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("label", action="store", type=str)
    parser.add_argument("--cache-dir", action="store", type=str, default=".")
    args = parser.parse_args()
    cache_filename = os.path.join(args.cache_dir, f"cache-{args.label}.npz")
    cache_filename_X_packer = os.path.join(args.cache_dir, f"X_packer-{args.label}.yml")
    cache_filename_y_packer = os.path.join(args.cache_dir, f"y_packer-{args.label}.yml")
    n_timestep_frequency = 4
    n_days_train = 8
    n_days_window = 2
    n_days_between_window = 1
    n_timesteps = int(4 * 24 * n_days_train // n_timestep_frequency)
    n_window = int(4 * 24 * n_days_window // n_timestep_frequency)
    n_between_windows = int(4 * 24 * n_days_between_window / n_timestep_frequency)
    units = 128
    n_hidden_layers = 3
    batch_size = 48
    input_variables = [
        "surface_geopotential",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "land_sea_mask",
        "total_precipitation",
        "latent_heat_flux",
        "sensible_heat_flux",
    ]
    prognostic_variables = [
        "air_temperature",
        "specific_humidity",
    ]
    if os.path.isfile(cache_filename):
        with open(cache_filename, "rb") as f:
            data = np.load(f)
            X_coarse = data["X_coarse"]
            y_coarse = data["y_coarse"]
            y_ref = data["y_ref"]
        with open(cache_filename_X_packer, "r") as f:
            X_packer = fv3fit.ArrayPacker.load(f)
        with open(cache_filename_y_packer, "r") as f:
            y_packer = fv3fit.ArrayPacker.load(f)
        assert "air_temperature" in y_packer.feature_counts
    else:
        X_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, input_variables)
        y_packer = fv3fit.ArrayPacker(SAMPLE_DIM_NAME, prognostic_variables)
        nudged_mapper = loaders.mappers.open_merged_nudged(
            COARSE_DATA_DIR,
            merge_files=("before_dynamics.zarr", "nudging_tendencies.zarr"),
        )
        ref_mapper = loaders.mappers.CoarsenedDataMapper(REF_DATA_DIR)
        X_coarse, y_coarse, y_ref = load_data(
            nudged_mapper,
            ref_mapper,
            X_packer,
            y_packer,
            n_timesteps=n_timesteps,
            n_timestep_frequency=n_timestep_frequency,
            label=args.label,
            cache_dir=args.cache_dir,
        )
        print("shuffling")
        # shuffle columns
        idx_shuffle = np.arange(X_coarse.shape[0])
        np.random.shuffle(idx_shuffle)
        X_coarse = X_coarse[idx_shuffle, :]
        y_coarse = y_coarse[idx_shuffle, :]
        y_ref = y_ref[idx_shuffle, :]
        if len(X_packer.feature_counts) == 0:
            raise RuntimeError(
                "need to process at least one file to store feature counts, "
                "please delete a file from the cache"
            )
        with open(cache_filename, "wb") as f:
            data = {
                "X_coarse": X_coarse,
                "y_coarse": y_coarse,
                "y_ref": y_ref,
            }
            print("saving")
            np.savez(f, **data)
        with open(cache_filename_X_packer, "w") as f:
            X_packer.dump(f)
        with open(cache_filename_y_packer, "w") as f:
            y_packer.dump(f)
    X_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler()
    X_scaler.fit(X_coarse)
    y_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler()
    y_scaler.fit(y_ref)
    X_train = WindowedDataSequence(
        X_coarse,
        y_coarse,
        y_ref,
        n_window=n_window,
        n_between_windows=n_between_windows,
        batch_size=batch_size,
    )
    print(X_coarse.shape)
    loss = fv3fit.keras._models.loss.get_weighted_mse(y_packer, y_scaler.std)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model = build_model(
        batch_size,
        X_coarse.shape[-1],
        y_coarse.shape[-1],
        n_window,
        units,
        n_hidden_layers,
        X_scaler,
        y_scaler,
    )
    model.compile(
        optimizer=optimizer, loss=loss,
    )
    model.fit(x=[X_coarse, y_coarse], y=y_ref, batch_size=48, epochs=100)
    # model.fit(x=X_train)

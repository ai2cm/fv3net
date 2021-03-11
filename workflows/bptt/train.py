from typing import Iterable, Hashable, Sequence, Tuple
import fv3gfs.util
import numpy as np
import fv3fit
import xarray as xr
import tensorflow as tf
import argparse
import concurrent.futures
from preprocessing import TrainingArrays
import copy
import random
import vcm

# sample_dim_name is arbitrary, just needs to be the same in each packer and
# in our model object
SAMPLE_DIM_NAME = "sample"


def get_packed_array(
    dataset: xr.Dataset, names: Iterable[Hashable],
) -> Tuple[np.ndarray, Tuple[int]]:
    """
    Transform dataset into packed numpy array.

    Requires the dataset be a "sample" dataset with dimensions [sample]
    or [sample, feature] for each variable.

    Args:
        dataset: data to transform
        names: variable names to put in packed array, in order
    
    Returns:
        packed: packed numpy array
        features: number of features corresponding to each of the input names
    """
    array_list = []
    features_list = []
    for name in names:
        if len(dataset[name].dims) == 1:  # scalar sample array
            array_list.append(dataset[name].values[:, None])
        elif len(dataset[name].dims) == 2:  # vertically-resolved sample array
            array_list.append(dataset[name].values[:, :])
        else:
            raise NotImplementedError(
                "received array with unexpected number of dimensions: "
                f"{dataset[name].dims}"
            )
    return np.concatenate(array_list, axis=-1), tuple(features_list)


def unpack_array(
    array: np.ndarray,
    names: Iterable[Hashable],
    features: Iterable[int],
    feature_dim_names: Iterable[str],
    sample_dim_name: str,
) -> xr.Dataset:
    """
    Args:
        array: a [sample, feature] array to unpack
        names: the name of each variable to unpack
        features: the number of features for each variable in names
        feature_dim_names: the names for the feature dimensions
        sample_dim_name: the name for the sample dimension in the output dataset
    """
    data_arrays = {}
    i_feature = 0
    for name, n_features, feature_dim_name in zip(names, features, feature_dim_names):
        if n_features == 1:
            data_arrays[name] = xr.DataArray(
                array[:, i_feature], dims=[sample_dim_name]
            )
        else:
            data_arrays[name] = xr.DataArray(
                array[:, i_feature : i_feature + n_features],
                dims=[sample_dim_name, feature_dim_name],
            )
        i_feature += n_features
    return xr.Dataset(data_arrays)


def build_model(
    n_input,
    n_state,
    n_window,
    units,
    n_hidden_layers,
    tendency_ratio,
    kernel_regularizer,
    timestep_seconds,
):
    forcing_input = tf.keras.layers.Input(shape=[n_window, n_input])
    given_tendency_input = tf.keras.layers.Input(shape=[n_window, n_state])
    state_delta = tf.keras.layers.Lambda(
        lambda x: x * tf.constant(timestep_seconds, dtype=tf.float32)
    )(given_tendency_input)

    rnn = tf.keras.layers.RNN(
        fv3fit.keras.GCMCell(
            units=units,
            n_input=n_input,
            n_state=n_state,
            n_hidden_layers=n_hidden_layers,
            tendency_ratio=tendency_ratio,
            dropout=0.0,
            kernel_regularizer=kernel_regularizer,
            use_spectral_normalization=False,
        ),
        name="rnn",
        return_sequences=True,
    )
    outputs = rnn(inputs=(forcing_input, state_delta))
    model = tf.keras.Model(
        inputs=[forcing_input, given_tendency_input], outputs=outputs
    )
    return model


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


class PreloadedArrayIterator:
    """
    Iterator for array data which asynchronously pre-loads
    the next set of arrays.
    """

    def __init__(
        self, filenames: Sequence[str],
    ):
        """
        Args:
            filenames: npz files containing TrainingArrays data
        """
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
                load_arrays, self.filenames[self._idx],
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


def load_arrays(filename):
    fs = vcm.get_fs(filename)
    with fs.open(filename, "rb") as f:
        return TrainingArrays.load(f)


def get_dim_lengths(arrays):
    n_input = arrays.inputs_baseline.shape[2]
    n_state = arrays.prognostic_baseline.shape[2]
    n_window = arrays.prognostic_baseline.shape[1]
    if arrays.prognostic_baseline.shape[1] != arrays.inputs_baseline.shape[1]:
        raise ValueError(
            f"array data has {arrays.prognostic_baseline.shape[1]} time points"
            f" for prognostic data and {arrays.inputs_baseline.shape[1]} time points"
            " for input data"
        )
    return n_input, n_state, n_window


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "model_output_dir", type=str, help="directory to write the trained model"
    )
    return parser


def get_stepwise_model(recurrent_model):
    forcing_input = tf.keras.layers.Input(shape=[n_input])
    state_input = tf.keras.layers.Input(shape=[n_state])
    # tendency is already incorporated into passed state
    tendencies = tf.keras.layers.Lambda(lambda x: x * 0.0)(state_input)
    gcm_cell = recurrent_model.get_layer("rnn").cell
    _, latent_state = gcm_cell([forcing_input, tendencies], [state_input])
    outputs = latent_state[0]
    model = tf.keras.Model(inputs=[forcing_input, state_input], outputs=outputs)
    return model


def prepare_keras_arrays(
    arrays, input_scaler, tendency_scaler, prognostic_scaler, timestep_seconds
):
    norm_input = input_scaler.normalize(arrays.inputs_baseline)
    norm_given_tendency = tendency_scaler.normalize(arrays.given_tendency)
    norm_prognostic_reference = prognostic_scaler.normalize(arrays.prognostic_reference)
    # use tendency to set model initial state on first timestep
    norm_given_tendency[:, 0, :] = (
        prognostic_scaler.normalize(arrays.prognostic_baseline)[:, 0, :]
        / timestep_seconds
    )
    return norm_input, norm_given_tendency, norm_prognostic_reference


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    parser = get_parser()
    args = parser.parse_args()
    fs = vcm.get_fs(args.arrays_dir)
    first_filename = fs.listdir(args.arrays_dir, detail=False)[0]
    with open(first_filename, "rb") as f:
        arrays = TrainingArrays.load(f)
        n_input, n_state, n_window = get_dim_lengths(arrays)
        prognostic_scaler = fv3fit.StandardScaler()
        prognostic_scaler.fit(arrays.prognostic_baseline)
        # do not remove mean for tendency scaling
        tendency_scaler = copy.deepcopy(prognostic_scaler)
        tendency_scaler.mean[:] = 0.0
        assert not np.all(prognostic_scaler.mean[:] == 0.0)
        input_scaler = fv3fit.StandardScaler()
        input_scaler.fit(arrays.inputs_baseline)
        input_packer = fv3fit.ArrayPacker(
            sample_dim_name=SAMPLE_DIM_NAME, pack_names=arrays.input_names,
        )
        # we have a hard-coded assumption that output state is
        # [specific_humidity, air_temperature] and that both have the same nz
        nz = int(arrays.prognostic_baseline.shape[2] / 2.0)
        prognostic_packer = fv3fit.ArrayPacker(
            sample_dim_name=SAMPLE_DIM_NAME,
            pack_names=("air_temperature", "specific_humidity"),
        )
        prognostic_packer._n_features = {
            "air_temperature": nz,
            "specific_humidity": nz,
        }
        prognostic_packer._dims = {
            "air_temperature": (SAMPLE_DIM_NAME, fv3gfs.util.Z_DIM),
            "specific_humidity": (SAMPLE_DIM_NAME, fv3gfs.util.Z_DIM),
        }
        input_names = arrays.input_names

    units = 32
    n_hidden_layers = 3
    tendency_ratio = 0.2  # scaling between values and their tendencies
    kernel_regularizer = tf.keras.regularizers.l2(0.001)
    timestep_seconds = 3 * 60 * 60
    # batch size to use for training
    batch_size = 48

    weight = np.zeros_like(prognostic_scaler.std)
    weight[:nz] = 0.5 * prognostic_scaler.std[:nz] / np.sum(prognostic_scaler.std[:nz])
    weight[nz:] = 0.5 * prognostic_scaler.std[nz:] / np.sum(prognostic_scaler.std[nz:])
    weight = tf.constant(weight.astype(np.float32), dtype=tf.float32)

    def loss(y_true, y_pred):
        return tf.math.reduce_sum(
            tf.reduce_mean(
                (weight[None, None, :] * tf.math.square(y_pred - y_true)), axis=(0, 1)
            )
        )

    # loss = penalize_negative_water(
    #     loss,
    #     1.0,
    #     -1.0 * prognostic_scaler.mean[nz:] / prognostic_scaler.std[nz:],
    # )
    optimizer = tf.keras.optimizers.Adam(lr=0.001, amsgrad=True, clipnorm=1.0)

    # this model does not normalize, it acts on normalized data
    model = build_model(
        n_input,
        n_state,
        n_window,
        units,
        n_hidden_layers,
        tendency_ratio,
        kernel_regularizer,
        timestep_seconds,
    )
    model.compile(
        optimizer=optimizer, loss=loss,
    )

    filenames = fs.listdir(args.arrays_dir, detail=False)
    train_filenames = filenames[:-4]
    validation_filename = filenames[-1]
    with fs.open(validation_filename, "rb") as f:
        validation_arrays = TrainingArrays.load(f)

    training_arrays = PreloadedArrayIterator(fs.listdir(args.arrays_dir, detail=False))

    val_inputs, val_given_tendency, val_target_out = prepare_keras_arrays(
        validation_arrays,
        input_scaler,
        tendency_scaler,
        prognostic_scaler,
        timestep_seconds,
    )
    val_base_out = timestep_seconds * np.cumsum(val_given_tendency, axis=1)
    baseline_loss = np.mean(
        loss(val_base_out.astype(np.float32), val_target_out.astype(np.float32))
    )
    print(f"baseline loss {baseline_loss}")

    base_epoch = 0
    for i_epoch in range(20):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        for arrays in training_arrays:
            (
                norm_input,
                norm_given_tendency,
                norm_prognostic_reference,
            ) = prepare_keras_arrays(
                arrays,
                input_scaler,
                tendency_scaler,
                prognostic_scaler,
                timestep_seconds,
            )
            model.fit(
                x=[norm_input, norm_given_tendency],
                y=[norm_prognostic_reference],
                batch_size=batch_size,
                epochs=1,
                shuffle=True,
            )
        val_out = model.predict([val_inputs, val_given_tendency])
        val_loss = np.mean(
            loss(val_out.astype(np.float32), val_target_out.astype(np.float32))
        )
        print(f"validation loss: {val_loss}")
    stepwise_model = get_stepwise_model(model)
    stepwise_model.compile(
        optimizer=optimizer, loss=loss,
    )
    recurrent = fv3fit.keras.RecurrentModel(
        SAMPLE_DIM_NAME,
        input_variables=list(input_names),
        model=stepwise_model,
        input_packer=input_packer,
        prognostic_packer=prognostic_packer,
        input_scaler=input_scaler,
        prognostic_scaler=prognostic_scaler,
        train_timestep_seconds=timestep_seconds,
    )
    fv3fit.dump(recurrent, args.model_output_dir)
    loaded = fv3fit.load(args.model_output_dir)

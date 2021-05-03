import argparse
import fsspec
import logging
import os
import pickle
import yaml
import tensorflow as tf
import xarray as xr
from toolz.functoolz import compose_left
from dataclasses import dataclass, asdict

from fv3fit._shared._transforms import (
    ArrayStacker, extract_ds_arrays, stack_io, standardize, unstandardize
)
from loaders.batches import shuffle
from loaders.batches._sequences import Map


# TODO Mesh this with usual training config
@dataclass
class TrainingConfig:
    train_data_path: str
    test_data_path: str
    input_variables: List[str]
    output_variables: List[str]
    save_path: str
    vertical_subselections: Mapping[str, List[int]] = None


logger = logging.getLogger(__name__)


# TODO use batch_func, batch_kwargs
def get_subsampled_batches(path):
    fs = fsspec.get_fs_token_paths(path)[0]
    files = fs.glob(os.path.join(path, "*.nc"))
    return Map(xr.open_dataset, files)


def _init_vert_subselection_slices_from_config(layer_limits):
    new_slice_limits = {
        k: slice(*v)
        for k, v in layer_limits.items()
    }
    return new_slice_limits


def process_vert_subselect_config(config):
    subselect_key = "vertical_subselections"
    if subselect_key in config:
        from_yaml = config[subselect_key]
        with_slices = _init_vert_subselection_slices_from_config(from_yaml)
        config.update({subselect_key: with_slices})

    return config


def _get_std_info(batches):
    # Currently uses profile max std as the scaling factor
    ignore_vars = ["cos_day", "sin_day", "cos_month", "sin_month"]
    sample_ds = xr.concat([b for b in shuffle(batches, seed=89)[:96]], "init")
    std_info = {}
    for var, da in sample_ds.items():
        if var not in ignore_vars:
            mean = da.mean(dim=["init", "sample"]).values + 0
            std = da.std(dim=["init", "sample"]).max().values + 0
        else:
            mean = 0
            std = 1
        std_info[var] = (mean, std)

    return std_info


def _save_preprocessing(path, X_stacker, y_stacker, std_info):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "X_stacker.yaml"), "w") as f:
        X_stacker.dump(f)
    with open(os.path.join(path, "y_stacker.yaml"), "w") as f:
        y_stacker.dump(f)
    with open(os.path.join(path, "standardization_info.pkl"), "wb") as f:
        pickle.dump(std_info, f)


def load_batch_preprocessing_chain(config: TrainingConfig, batches):
    # func that goes from dataset -> X, y
    std_info = _get_std_info(batches)

    peeked = batches[0]
    X_stacker = ArrayStacker.from_data(
        peeked,
        config.input_variables,
        feature_subselections=config.vertical_subselections
    )
    y_stacker = ArrayStacker.from_data(
        peeked,
        config.output_variables,
        feature_subselections=config.vertical_subselections
    )

    stack_func = stack_io(X_stacker, y_stacker)
    std_func = standardize(std_info)
    preproc = compose_left(extract_ds_arrays, std_func, stack_func)

    _save_preprocessing(config.save_path, X_stacker, y_stacker, std_info)

    return preproc


def get_batch_generator_constructer(preproc_func, batches):
    def get_generator():
        for batch in batches:
            X, y = preproc_func(batch)
            X = tf.convert_to_tensor(X)
            y = tf.convert_to_tensor(y)
            yield tf.data.Dataset.from_tensor_slices((X, y))
            
    return get_generator


def create_tf_dataset_for_training(gen_create_func):
    peeked = next(gen_create_func())
    output_sig = tf.data.DatasetSpec.from_value(peeked)
    tf_ds = tf.data.Dataset.from_generator(
        gen_create_func,
        output_signature=output_sig
    )

    return tf_ds


def batches_to_tf_dataset(preproc_func, batches):
    get_generator = get_batch_generator_constructer(preproc_func, batches)
    tf_ds = create_tf_dataset_for_training(get_generator)

    return tf_ds.prefetch(tf.data.AUTOTUNE).interleave(lambda x: x)


def get_emu_model(X, y):
    # TODO: Just replicate notebook for now
    inputs = tf.keras.layers.Input(X.shape[-1])
    hidden_layer = tf.keras.layers.Dense(
        250, activation=tf.keras.activations.tanh
    )(inputs)
    outputs = tf.keras.layers.Dense(y.shape[-1])(hidden_layer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [20000, 300000], [1e-3, 1e-4, 1e-5]
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


def save_model(config, model):
    
    # Remove compiled specialized metrics because these bork loading
    model.optimizer = None
    model.compiled_loss = None
    model.compiled_metrics = None

    model.save(os.path.join(config.save_path, "model.tf"), save_format="tf")
    with open(os.path.join(config.save_path, "model_options.yaml"), "w") as f:
        yaml.dump(
            dict(
                input_variables=config.input_variables,
                output_variables=config.output_variables,
                sample_dim_name="sample"
            ),
            f
        )



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    with open(args.train_config, "rb") as f:
        config_yaml = yaml.safe_load(f)

    config_yaml = process_vert_subselect_config(config_yaml)
    config = TrainingConfig(**config_yaml)

    train_batches = get_subsampled_batches(config.train_data_path)
    test_batches = get_subsampled_batches(config.test_data_path)
    preproc_chain = load_batch_preprocessing_chain(config, train_batches)

    train_ds = batches_to_tf_dataset(preproc_chain, train_batches)
    test_ds = batches_to_tf_dataset(preproc_chain, test_batches)

    X, y = preproc_chain(train_batches[0])
    model = get_emu_model(X, y)

    model.fit(
        train_ds.shuffle(100000).batch(128),
        epochs=50,
        validation_data=test_ds.batch(128),
        validation_freq=5,
        verbose=2,
    )

    save_model(config, model)

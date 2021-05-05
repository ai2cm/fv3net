import argparse
import datetime
import fsspec
import logging
import os
import pickle
import tempfile
import yaml
import tensorflow as tf
import xarray as xr
from dataclasses import dataclass
from toolz.functoolz import compose_left
from typing import Any, List, Mapping, MutableMapping, Union

from fv3fit._shared._transforms import (
    ArrayStacker, extract_ds_arrays, stack_io, standardize
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
    vertical_subselections: Union[Mapping[str, List[int]], None] = None
    fit_kwargs: Union[MutableMapping[str, Any], None] = None


logger = logging.getLogger(__name__)


# TODO use batch_func, batch_kwargs from usual training config?
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


def load_config(config_path):

    with fsspec.open(config_path, "rb") as f:
        config_yaml = yaml.safe_load(f)

    config_yaml = process_vert_subselect_config(config_yaml)
    return TrainingConfig(**config_yaml)


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

    save_info = dict(X_stacker=X_stacker, y_stacker=y_stacker, std_info=std_info)

    return preproc, save_info, y_stacker._stacked_feature_indices


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


def get_fit_kwargs(config):
    # awkward right now because need batch size for tf_dataset
    if config.fit_kwargs is None:
        fit_kwargs = {}
    else:
        # TODO: should this be copy/deep copy?
        fit_kwargs = dict(**config.fit_kwargs)

    batch_size = fit_kwargs.pop("batch_size", 128)

    return batch_size,  fit_kwargs


class ByVariableMSE(tf.keras.metrics.MeanSquaredError, tf.keras.metrics.Metric):
    
    def __init__(self, name, var_slice, **kwargs):
        super().__init__(name=name, **kwargs)
        self._slice = var_slice
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[..., self._slice]
        y_pred = y_pred[..., self._slice]
        if sample_weight is not None:
            sample_weight = sample_weight[..., self._slice]
            
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
        
        
def create_by_var_mse(var_slices):
    
    metrics = [
        ByVariableMSE(varname, vslice)
        for varname, vslice in var_slices.items()
    ]
    return metrics


def get_emu_model(X, y, y_feature_indices=None):
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

    if y_feature_indices is not None:
        extra_metrics = create_by_var_mse(y_feature_indices)
    else:
        extra_metrics = []

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"]+extra_metrics)

    return model


def fit_model(config, model):
    tmpdir = tempfile.TemporaryDirectory()
    log_dir = os.path.join(
        tmpdir.name,
        "tensorboard_log-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq=1000
    )

    batch_size, fit_kwargs = get_fit_kwargs(config)
    
    model.fit(
        train_ds.shuffle(100000).batch(batch_size),
        validation_data=test_ds.batch(batch_size),
        callbacks=[tensorboard_callback],
        **fit_kwargs
    )

    return tmpdir


def _save_model(path, model, X_stacker, y_stacker, std_info):
    with open(os.path.join(path, "X_stacker.yaml"), "w") as f:
        X_stacker.dump(f)
    with open(os.path.join(path, "y_stacker.yaml"), "w") as f:
        y_stacker.dump(f)
    with open(os.path.join(path, "standardization_info.pkl"), "wb") as f:
        pickle.dump(std_info, f)
    
    # Remove compiled specialized metrics because these bork loading
    model.optimizer = None
    model.compiled_loss = None
    model.compiled_metrics = None

    model.save(os.path.join(path, "model.tf"), save_format="tf")


def _save_to_destination(source, destination):

    fs = fsspec.get_fs_token_paths(destination)[0]
    fs.makedirs(destination, exist_ok=True)
    fs.put(source, destination, recursive=True)


def save(path, model, logdir, X_stacker, y_stacker, std_info):

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_to_destination(logdir, tmpdir)
        _save_model(tmpdir, model, X_stacker, y_stacker, std_info)
        _save_to_destination(tmpdir, path)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = load_config(args.train_config)

    train_batches = get_subsampled_batches(config.train_data_path)
    test_batches = get_subsampled_batches(config.test_data_path)
    preproc_chain, preproc_save_info, y_features = \
        load_batch_preprocessing_chain(config, train_batches)
    train_ds = batches_to_tf_dataset(preproc_chain, train_batches)
    test_ds = batches_to_tf_dataset(preproc_chain, test_batches)

    X, y = preproc_chain(train_batches[0])
    model = get_emu_model(X, y, y_feature_indices=y_features)
    fit_logdir = fit_model(config, model)

    save(config.save_path, model, fit_logdir.name, **preproc_save_info)

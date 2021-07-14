import argparse
import datetime
import fsspec
import logging
import os
import pickle
import tempfile
import yaml
import tensorflow as tf
from dataclasses import dataclass
from typing import Any, MutableMapping, Union

from fv3fit.emulation.data import nc_dir_to_tf_dataset, TransformConfig


# TODO Mesh this with usual training config
@dataclass
class TrainingConfig:
    train_data_path: str
    test_data_path: str
    transform_config: TransformConfig
    save_path: str
    fit_kwargs: Union[MutableMapping[str, Any], None] = None


logger = logging.getLogger(__name__)


def load_config(config_path):

    with fsspec.open(config_path, "rb") as f:
        config_yaml = yaml.safe_load(f)

    return TrainingConfig(**config_yaml)


def get_fit_kwargs(config):
    # awkward right now because need batch size for tf_dataset
    if config.fit_kwargs is None:
        fit_kwargs = {}
    else:
        # TODO: should this be copy/deep copy?
        fit_kwargs = dict(**config.fit_kwargs)

    batch_size = fit_kwargs.pop("batch_size", 128)

    return batch_size, fit_kwargs


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


def _save_model(
    path, model,
    std_info
):
    # TODO: currently legacy inputs from the AllPhysics .load
    model_options = {
    }

    with open(os.path.join(path, "standardization_info.pkl"), "wb") as f:
        pickle.dump(std_info, f)
    with open(os.path.join(path, "model_options.yaml"), "w") as f:
        f.write(yaml.dump(model_options))
    
    # Remove compiled specialized metrics because these bork loading
    model.optimizer = None
    model.compiled_loss = None
    model.compiled_metrics = None

    model.save(os.path.join(path, "model.tf"), save_format="tf")


def _save_to_destination(source, destination):

    fs = fsspec.get_fs_token_paths(destination)[0]
    fs.makedirs(destination, exist_ok=True)
    fs.put(source, destination, recursive=True)


def save(path, model, logdir, config, X_stacker, y_stacker, std_info):

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
    preproc_config = config.data_preprocess_config

    train_batches = get_subsampled_batches(config.train_data_path)
    test_batches = get_subsampled_batches(config.test_data_path)
    preproc_chain = load_input_transforms(config.data_preprocess_config)
    train_ds = batches_to_tf_dataset(preproc_chain, train_batches)
    test_ds = batches_to_tf_dataset(preproc_chain, test_batches)

    X, y = preproc_chain(train_batches[0])
    model = get_emu_model(
        X,
        y,
        preproc_config.input_variables,
        preproc_config.output_variables
    )
    fit_logdir = fit_model(config, model)

    save(config.save_path, model, fit_logdir.name, config, **preproc_save_info)

import argparse
import os
import yaml
import logging
import numpy as np
import sys
import random
import joblib
from . import get_model
from ._validation_data import (
    check_validation_train_overlap,
    validation_timesteps_config,
)
from ._save_history import save_history
from .. import _shared as shared
import fv3fit._shared.io
import loaders
import tensorflow as tf
from typing import Union, Optional
import xarray as xr


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger(__file__)


MODEL_FILENAME = "model_data"
HISTORY_OUTPUT_DIR = "training_history"


def _get_optimizer(hyperparameters: dict = None):
    hyperparameters = hyperparameters or {}
    optimizer_config = hyperparameters.pop("optimizer", {})
    if optimizer_config:
        optimizer_class = getattr(
            tf.keras.optimizers, optimizer_config.get("name", "Adam")
        )
        learning_rate = optimizer_config.get("learning_rate", None)
        optimizer = (
            optimizer_class(learning_rate=learning_rate)
            if learning_rate is not None
            else optimizer_class()
        )
    else:
        optimizer = None
    return optimizer


def _set_random_seed(seed: Union[float, int] = 0):
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    tf.random.set_seed(seed + 3)


def _validation_dataset(
    train_config: shared.ModelTrainingConfig,
) -> Optional[xr.Dataset]:
    if len(train_config.validation_timesteps) > 0:
        check_validation_train_overlap(
            train_config.batch_kwargs["timesteps"], train_config.validation_timesteps,
        )
        validation_config = validation_timesteps_config(train_config)
        # validation config puts all data in one batch
        validation_dataset_sequence = shared.load_data_sequence(
            data_path, validation_config
        )
        if len(validation_dataset_sequence) > 1:
            raise ValueError(
                "Something went wrong! "
                "All validation data should be concatenated into a single batch. There "
                f"are {len(validation_dataset_sequence)} batches in the sequence."
            )
        return validation_dataset_sequence[0]
    else:
        validation_dataset = None
    return validation_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_path", nargs="*", type=str, help="Location of training data"
    )
    parser.add_argument(
        "train_config_file",
        type=str,
        help="Local path for training configuration yaml file",
    )
    parser.add_argument(
        "output_data_path", type=str, help="Location to save config and trained model."
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="json file containing a list of timesteps in YYYYMMDD.HHMMSS format",
    )
    parser.add_argument(
        "--validation-timesteps-file",
        type=str,
        default=None,
        help="json file containing a list of validation timesteps in "
        "YYYYMMDD.HHMMSS format",
    )
    parser.add_argument(
        "--local-download-path",
        type=str,
        help="Optional path for downloading data before training. If not provided, "
        "will read from remote every epoch. Local download greatly speeds NN training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = shared.parse_data_path(args)
    train_config = shared.load_model_training_config(
        args.train_config_file, args.train_data_path
    )

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_config.batch_kwargs["timesteps"] = timesteps
    if args.validation_timesteps_file:
        with open(args.validation_timesteps_file, "r") as f:
            val_timesteps = yaml.safe_load(f)
        train_config.validation_timesteps = val_timesteps
    shared.save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)
    _set_random_seed(train_config.random_seed)
    optimizer = _get_optimizer(train_config.hyperparameters)

    fit_kwargs = train_config.hyperparameters.pop("fit_kwargs", {})
    model = get_model(
        train_config.model_type,
        loaders.SAMPLE_DIM_NAME,
        train_config.input_variables,
        train_config.output_variables,
        optimizer=optimizer,
        **train_config.hyperparameters,
    )
    batches = shared.load_data_sequence(data_path, train_config)
    if args.local_download_path:
        batches = batches.local(args.local_download_path)
        # joblib.Memory uses joblib to cache data to disk
        memory = joblib.Memory(args.local_download_path)
        batches.__getitem__ = memory.cache(batches.__getitem__)

    validation_dataset = _validation_dataset(train_config)

    history = model.fit(batches, validation_dataset, **fit_kwargs)  # type: ignore
    fv3fit._shared.io.dump(model, args.output_data_path)
    save_history(
        history, os.path.join(args.output_data_path, HISTORY_OUTPUT_DIR),
    )

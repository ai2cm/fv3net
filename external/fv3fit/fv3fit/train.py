import argparse
from importlib import import_module
import loaders
import logging
import os
import yaml

from ._shared import (
    save_config_output,
    parse_data_path,
    load_model_training_config,
    load_data_sequence,
    ModelTrainingConfig,
    io,
)
from .keras._training import get_regularizer, get_optimizer, set_random_seed
from .keras._validation_data import validation_dataset


KERAS_CHECKPOINT_PATH = "model_checkpoints"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "routine",
        type=str,
        help="Training routine to use. Choices are 'keras' or 'sklearn'",
    )
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
        help="Json file containing a list of validation timesteps in "
        "YYYYMMDD.HHMMSS format. Only relevant for keras training",
    )
    parser.add_argument(
        "--local-download-path",
        type=str,
        help="Optional path for downloading data before training. If not provided, "
        "will read from remote every epoch. Local download greatly speeds NN training.",
    )
    return parser.parse_args()


def _get_model_kwargs(routine: str, config: ModelTrainingConfig) -> dict:
    # returns the args for the appropriate training routine's get_model
    if routine == "sklearn":
        return {"train_config": config}
    elif routine == "keras":
        return {
            "model_type": config.model_type,
            "sample_dim_name": loaders.SAMPLE_DIM_NAME,
            "input_variables": config.input_variables,
            "output_variables": config.output_variables,
            "optimizer": get_optimizer(config.hyperparameters),
            "kernel_regularizer": get_regularizer(config.hyperparameters),
            "checkpoint_path": (
                os.path.join(args.output_data_path, KERAS_CHECKPOINT_PATH)
                if config.save_model_checkpoints
                else None
            ),
            **config.hyperparameters,
        }
    else:
        raise ValueError("arg for 'routine' should be either 'sklearn' or 'keras'.")


def _keras_fit_kwargs(config: ModelTrainingConfig) -> dict:
    # extra args specific to keras training
    fit_kwargs = config.hyperparameters.pop("fit_kwargs", {})
    fit_kwargs["validation_dataset"] = validation_dataset(data_path, train_config)
    return fit_kwargs


if __name__ == "__main__":
    args = parse_args()
    data_path = parse_data_path(args)
    train_config = load_model_training_config(
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

    save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)

    if args.routine == "keras":
        set_random_seed(train_config.random_seed)
        fit_kwargs = _keras_fit_kwargs(train_config)
    else:
        fit_kwargs = {}

    batches = load_data_sequence(data_path, train_config)
    if args.local_download_path:
        batches = batches.local(args.local_download_path)  # type: ignore

    get_model = getattr(import_module(f"fv3fit.{args.routine}"), "get_model")
    model_kwargs = _get_model_kwargs(args.routine, train_config)
    model = get_model(**model_kwargs)
    model.fit(batches, **fit_kwargs)
    io.dump(model, args.output_data_path)

import argparse
import loaders
import logging
import os
import xarray as xr
from typing import Sequence, Optional
import yaml
import dataclasses
import fsspec

from fv3fit._shared import (
    parse_data_path,
    load_data_sequence,
    io,
    Estimator,
)
import fv3fit._shared.config
from .keras._training import get_regularizer, get_optimizer, set_random_seed
import fv3fit.keras
import fv3fit.sklearn
import fv3fit


KERAS_CHECKPOINT_PATH = "model_checkpoints"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", nargs="*", type=str, help="Location of training data"
    )
    parser.add_argument(
        "config_file", type=str, help="Local path for training configuration yaml file",
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
        "will read from remote every epoch. Local download greatly speeds training.",
    )
    return parser


def _get_model(config: fv3fit.TrainingConfig) -> Estimator:
    if isinstance(config, fv3fit.SklearnTrainingConfig):
        return fv3fit.sklearn.get_model(
            model_type=config.model_type,
            input_variables=config.input_variables,
            output_variables=config.output_variables,
            scaler_type=config.scaler_type,
            scaler_kwargs=config.scaler_kwargs,
            **config.hyperparameters,
        )
    elif isinstance(config, fv3fit.KerasTrainingConfig):
        checkpoint_path = (
            os.path.join(args.output_data_path, KERAS_CHECKPOINT_PATH)
            if config.save_model_checkpoints
            else None
        )
        return fv3fit.keras.get_model(
            model_type=config.model_type,
            sample_dim_name=loaders.SAMPLE_DIM_NAME,
            input_variables=config.input_variables,
            output_variables=config.output_variables,
            optimizer=get_optimizer(config.hyperparameters),
            kernel_regularizer=get_regularizer(config.hyperparameters),
            checkpoint_path=checkpoint_path,
            **config.hyperparameters,
        )
    else:
        raise NotImplementedError(f"Model type {config.model_type} is not implemented.")


def dump_dataclass(obj, yaml_filename):
    with fsspec.open(yaml_filename, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    data_path = parse_data_path(args.data_path)
    (
        legacy_config,
        train_config,
        train_data_config,
        val_data_config,
    ) = fv3fit._shared.config.load_configs(
        args.config_file,
        data_path=data_path,
        output_data_path=args.output_data_path,
        timesteps_file=args.timesteps_file,
        validation_timesteps_file=args.validation_timesteps_file,
    )
    set_random_seed(train_config.random_seed)

    dump_dataclass(train_config, os.path.join(args.output_data_path, "train.yaml"))
    dump_dataclass(
        train_data_config, os.path.join(args.output_data_path, "training_data.yaml")
    )

    train_batches = load_data_sequence(train_data_config)
    if val_data_config is not None:
        dump_dataclass(
            val_data_config, os.path.join(args.output_data_path, "validation_data.yaml")
        )
        val_batches: Optional[Sequence[xr.Dataset]] = load_data_sequence(
            val_data_config
        )
    else:
        val_batches = None

    if args.local_download_path:
        train_batches = train_batches.local(
            os.path.join(args.local_download_path, "train")
        )
        # TODO: currently, validation data is actually ignored except for Keras
        # where it is handled in an odd way during configuration setup. Refactor
        # model fitting to take in validation data directly, so this val_batches
        # (or val_dataset if you need to refactor it to one) is actually used
        val_batches = train_batches.local(
            os.path.join(args.local_download_path, "validation")
        )

    model = _get_model(train_config)
    model.fit(train_batches)
    train_config.model_path = args.output_data_path
    io.dump(model, args.output_data_path)

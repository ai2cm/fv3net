import argparse
import logging
import os
from typing import Sequence
from fv3fit._shared.config import get_arg_updated_config_dict
import yaml
import dataclasses
import fsspec

import fv3fit.keras
import fv3fit.sklearn
import fv3fit
import xarray as xr
import loaders


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_config", type=str, help="path of fv3fit.TrainingConfig yaml file",
    )
    parser.add_argument(
        "training_data_config",
        type=str,
        help="path of loaders.BatchesLoader training data yaml file",
    )
    parser.add_argument(
        "output_path", type=str, help="path to save config and trained model"
    )
    parser.add_argument(
        "--validation-data-config",
        type=str,
        default=None,
        help=(
            "path of loaders.BatchesLoader validation data yaml file, "
            "by default an empty sequence is used"
        ),
    )
    parser.add_argument(
        "--local-download-path",
        type=str,
        help=(
            "optional path for downloading data before training, "
            "can greatly increase training speed"
        ),
    )
    return parser


def dump_dataclass(obj, yaml_filename):
    with fsspec.open(yaml_filename, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f)


def main(args, unknown_args=None):
    with open(args.training_config, "r") as f:
        config_dict = yaml.safe_load(f)
        if unknown_args is not None:
            config_dict = get_arg_updated_config_dict(
                args=unknown_args, config_dict=config_dict
            )
        training_config = fv3fit.TrainingConfig.from_dict(config_dict)

    with open(args.training_data_config, "r") as f:
        training_data_config = loaders.BatchesLoader.from_dict(yaml.safe_load(f))

    fv3fit.set_random_seed(training_config.random_seed)

    dump_dataclass(training_config, os.path.join(args.output_path, "train.yaml"))
    dump_dataclass(
        training_data_config, os.path.join(args.output_path, "training_data.yaml")
    )

    train_batches: loaders.typing.Batches = training_data_config.load_batches(
        variables=training_config.variables
    )
    if args.validation_data_config is not None:
        with open(args.validation_data_config, "r") as f:
            validation_data_config = loaders.BatchesLoader.from_dict(yaml.safe_load(f))
        dump_dataclass(
            validation_data_config,
            os.path.join(args.output_path, "validation_data.yaml"),
        )
        val_batches = validation_data_config.load_batches(
            variables=training_config.variables
        )
    else:
        val_batches: Sequence[xr.Dataset] = []

    if args.local_download_path:
        train_batches = loaders.to_local(
            train_batches, os.path.join(args.local_download_path, "train")
        )
        val_batches = loaders.to_local(
            val_batches, os.path.join(args.local_download_path, "validation")
        )

    train = fv3fit.get_training_function(training_config.model_type)
    model = train(
        hyperparameters=training_config.hyperparameters,
        train_batches=train_batches,
        validation_batches=val_batches,
    )
    if len(training_config.derived_output_variables) > 0:
        model = fv3fit.DerivedModel(model, training_config.derived_output_variables)
    fv3fit.dump(model, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    main(args, unknown_args)

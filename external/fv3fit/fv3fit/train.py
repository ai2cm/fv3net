import argparse
import logging
import os
from typing import Optional
import yaml
import dataclasses
import fsspec

from fv3fit._shared import parse_data_path, io, DerivedModel
import fv3fit._shared.config
import fv3fit.keras
import fv3fit.sklearn
import fv3fit
import loaders


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


def dump_dataclass(obj, yaml_filename):
    with fsspec.open(yaml_filename, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f)


def main(args):
    data_path = parse_data_path(args.data_path)
    (
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

    # TODO: uncomment this line when we aren't using fit_kwargs
    # to contain validation data
    # dump_dataclass(train_config, os.path.join(args.output_data_path, "train.yaml"))
    dump_dataclass(
        train_data_config, os.path.join(args.output_data_path, "training_data.yaml")
    )

    train_batches: loaders.typing.Batches = train_data_config.load_batches(
        variables=train_config.input_variables
        + train_config.output_variables
        + train_config.additional_variables
    )
    if val_data_config is not None:
        dump_dataclass(
            val_data_config, os.path.join(args.output_data_path, "validation_data.yaml")
        )
        val_batches: Optional[loaders.typing.Batches] = val_data_config.load_sequence()
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
        if val_batches is not None:
            val_batches = val_batches.local(
                os.path.join(args.local_download_path, "validation")
            )

    train = fv3fit.get_training_function(train_config.model_type)
    model = train(
        input_variables=train_config.input_variables,
        output_variables=train_config.output_variables,
        hyperparameters=train_config.hyperparameters,
        train_batches=train_batches,
        validation_batches=val_batches,
    )
    if len(train_config.derived_output_variables) > 0:
        model = DerivedModel(model, train_config.derived_output_variables,)
    io.dump(model, args.output_data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    main(args)

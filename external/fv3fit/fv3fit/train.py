import argparse
import logging
import os
import yaml
import dataclasses
import fsspec

import fv3fit.keras
import fv3fit.sklearn
import fv3fit
import loaders


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_config", type=str, help="path of fv3fit.TrainingConfig yaml file",
    )
    parser.add_argument(
        "train_data_config",
        type=str,
        help="path of loaders.BatchesLoader training data yaml file",
    )
    parser.add_argument(
        "val_data_config",
        type=str,
        help="path of loaders.BatchesLoader validation data yaml file",
    )
    parser.add_argument(
        "output_path", type=str, help="path to save config and trained model"
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


def main(args):
    with open(args.train_config, "r") as f:
        train_config = fv3fit.TrainingConfig.from_dict(yaml.load(f))
    with open(args.train_data_config, "r") as f:
        train_data_config = loaders.BatchesLoader.from_dict(yaml.load(f))
    with open(args.val_data_config, "r") as f:
        val_data_config = loaders.BatchesLoader.from_dict(yaml.load(f))

    fv3fit.set_random_seed(train_config.random_seed)

    dump_dataclass(train_config, os.path.join(args.output_path, "train.yaml"))
    dump_dataclass(
        train_data_config, os.path.join(args.output_path, "training_data.yaml")
    )

    all_variables = (
        train_config.input_variables
        + train_config.output_variables
        + train_config.additional_variables
    )
    train_batches: loaders.typing.Batches = train_data_config.load_batches(
        variables=all_variables
    )
    dump_dataclass(
        val_data_config, os.path.join(args.output_path, "validation_data.yaml")
    )
    val_batches = val_data_config.load_batches(variables=all_variables)

    if args.local_download_path:
        train_batches = train_batches.local(
            os.path.join(args.local_download_path, "train")
        )
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
        model = fv3fit.DerivedModel(model, train_config.derived_output_variables)
    fv3fit.dump(model, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    main(args)

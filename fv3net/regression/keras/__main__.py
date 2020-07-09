import argparse
import os
import yaml
import logging
from . import get_model
from .. import shared

# TODO: refactor these to ..shared
from ..sklearn.__main__ import _save_config_output


MODEL_FILENAME = "model_data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=str, help="Location of training data")
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
        "--no-train-subdir-append",
        action="store_true",
        help="Omit the appending of 'train' to the input training data path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_path = args.train_data_path
    if not args.no_train_subdir_append:
        data_path = os.path.join(data_path, "train")
    train_config = shared.load_model_training_config(args.train_config_file)

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_config.batch_kwargs["timesteps"] = timesteps

    _save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)

    model = get_model(
        train_config.model_type,
        train_config.input_variables,
        train_config.output_variables,
        **train_config.hyperparameters
    )
    batches = shared.load_data_sequence(data_path, train_config)
    model.fit(batches)

    model_output_path = os.path.join(args.output_data_path, MODEL_FILENAME)
    model.dump(model_output_path)

    report_metadata = {**vars(args), **vars(train_config)}

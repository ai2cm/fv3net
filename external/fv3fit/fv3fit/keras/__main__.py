import argparse
import os
import yaml
import logging
import sys
from . import get_model
from .. import _shared as shared
import loaders

# TODO: refactor these to ..shared
from ..sklearn.__main__ import _save_config_output

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger(__file__)


MODEL_FILENAME = "model_data"


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
        "--no-train-subdir-append",
        action="store_true",
        help="Omit the appending of 'train' to the input training data path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = shared.parse_data_path(args)
    train_config = shared.load_model_training_config(args.train_config_file)

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_config.batch_kwargs["timesteps"] = timesteps

    _save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)

    batch_size = train_config.hyperparameters.pop("batch_size", None)
    epochs = train_config.hyperparameters.pop("epochs", None)

    model = get_model(
        train_config.model_type,
        loaders.SAMPLE_DIM_NAME,
        train_config.input_variables,
        train_config.output_variables,
        **train_config.hyperparameters
    )
    batches = shared.load_data_sequence(data_path, train_config)
    model.fit(batches, batch_size, epochs)

    model_output_path = os.path.join(args.output_data_path, MODEL_FILENAME)
    model.dump(model_output_path)

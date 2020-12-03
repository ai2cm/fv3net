import argparse
import yaml
import logging
from . import _train as train
from .. import _shared as shared
import fv3fit._shared.io

TIMESTEPS_USED_FILENAME = "timesteps_used.yml"


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
        "--delete-local-results-after-upload",
        type=bool,
        default=True,
        help="If results are uploaded to remote storage, "
        "remove local copy after upload.",
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="json file containing a list of timesteps in YYYYMMDD.HHMMSS format",
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

    batched_data = shared.load_data_sequence(data_path, train_config)
    shared.save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)

    model = train.train_model(batched_data, train_config)
    fv3fit._shared.io.dump(model, args.output_data_path)

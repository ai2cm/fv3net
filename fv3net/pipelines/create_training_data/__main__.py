"""
Example of timesteps_file input data::

    {
        "train": [
            [ "20160801.003000", "20160801.004500" ],
            [ "20160801.001500", "20160801.003000" ]
        ],
        "test": [
            [ "20160801.011500", "20160801.013000" ],
            [ "20160801.010000", "20160801.011500" ]
        ]
    }⏎
"""
import argparse
import logging
import yaml
from .config import get_config

from .pipeline import run

example_timesteps_file = """
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog=__doc__)
    parser.add_argument(
        "--variable_namefile",
        type=str,
        default=None,
        help="yaml file for updating the default data variable names.",
    )
    parser.add_argument(
        "gcs_input_data_path",
        type=str,
        help="Location of input data in Google Cloud Storage bucket. "
        "Don't include bucket or big.zarr in path.",
    )
    parser.add_argument(
        "diag_c48_path",
        type=str,
        help="Directory containing diagnostic zarr directory coarsened from C384 "
        "for features (SHF, LHF, etc.) that are not saved in restarts.",
    )
    parser.add_argument(
        "timesteps_file",
        type=str,
        help=(
            "File containing paired timesteps for test and/or train sets. See the "
            "__doc__ of this script for more info."
        ),
    )
    parser.add_argument(
        "gcs_output_data_dir",
        type=str,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )

    args, pipeline_args = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)

    if args.variable_namefile is None:
        updates = {}
    else:
        with open(args.variable_namefile, "r") as f:
            updates = yaml.safe_load(f)
    names = get_config(updates)

    with open(args.timesteps_file, "r") as f:
        timesteps = yaml.safe_load(f)
    run(args=args, pipeline_args=pipeline_args, names=names, timesteps=timesteps)

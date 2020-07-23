from typing import Union, Sequence, Mapping
from datetime import datetime
import argparse
import os
import fsspec
import yaml
import logging
import report
import fv3viz as viz
import vcm
import numpy as np
from . import train
from .. import shared


MODEL_FILENAME = "sklearn_model.pkl"
MODEL_CONFIG_FILENAME = "training_config.yml"
TIMESTEPS_USED_FILENAME = "timesteps_used.yml"
REPORT_TITLE = "ML model training report"
TRAINING_FIG_FILENAME = "count_of_training_times_used.png"


def _save_config_output(output_url, config):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)

    with fs.open(config_url, "w") as f:
        yaml.dump(config, f)


def _create_report_plots(
    path: str, times: Sequence[Union[np.datetime64, datetime, str]]
) -> Mapping[str, Sequence[str]]:
    """Given path to output directory and times used, create all plots required
    for html report"""
    times = [vcm.cast_to_datetime(time) for time in times]
    with fsspec.open(os.path.join(path, TRAINING_FIG_FILENAME), "wb") as f:
        viz.plot_daily_and_hourly_hist(times).savefig(f, dpi=90)
    return {"Time distribution of training samples": [TRAINING_FIG_FILENAME]}


def _write_report(output_dir, sections, metadata, title):
    filename = title.replace(" ", "_") + ".html"
    html_report = report.create_html(sections, title, metadata=metadata)
    with fsspec.open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


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

    batched_data = shared.load_data_sequence(data_path, train_config)
    _save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)

    model = train.train_model(batched_data, train_config)
    train.save_model(args.output_data_path, model, MODEL_FILENAME)
    report_metadata = {**vars(args), **vars(train_config)}
    report_sections = _create_report_plots(
        args.output_data_path, batched_data.attrs["times"],
    )
    _write_report(args.output_data_path, report_sections, report_metadata, REPORT_TITLE)

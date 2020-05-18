import argparse
import os
import fsspec
import yaml
import logging
import report
import vcm
from . import train


MODEL_FILENAME = "sklearn_model.pkl"
MODEL_CONFIG_FILENAME = "training_config.yml"
TIMESTEPS_USED_FILENAME = "timesteps_used.yml"
REPORT_TITLE = "ML model training report"
TRAINING_FIG_FILENAME = "count_of_training_times_used.png"


def _save_config_output(output_url, config, timesteps):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)

    with fs.open(config_url, "w") as f:
        yaml.dump(config, f)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = os.path.join(args.train_data_path, "train")
    train_config = train.load_model_training_config(args.train_config_file)
    batched_data = train.load_data_sequence(data_path, train_config)
    _save_config_output(args.output_data_path, train_config)

    logging.basicConfig(level=logging.INFO)

    model = train.train_model(batched_data, train_config)
    train.save_model(args.output_data_path, model, MODEL_FILENAME)
    report_metadata = {**vars(args), **vars(train_config)}
    report_sections = {}
    _write_report(args.output_data_path, report_sections, report_metadata, REPORT_TITLE)

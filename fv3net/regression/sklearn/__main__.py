import argparse
import os
import fsspec
import yaml
import gallery
import report
import vcm
from .train import load_data_generator, load_model_training_config, train_model, save_model


MODEL_FILENAME = "sklearn_model.pkl"
MODEL_CONFIG_FILENAME = "training_config.yml"
TIMESTEPS_USED_FILENAME = "timesteps_used.yml"
REPORT_TITLE = "ML model training report"
TRAINING_FIG_FILENAME = "count_of_training_times_used.png"


def _save_config_output(output_url, config, timesteps):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)
    timesteps_url = os.path.join(output_url, TIMESTEPS_USED_FILENAME)

    with fs.open(config_url, "w") as f:
        yaml.dump(config, f)

    with fs.open(timesteps_url, "w") as f:
        yaml.dump(timesteps, f)


def _create_report_plots(path):
    """Given path to directory containing timesteps used, create all plots required
    for html report"""
    with fsspec.open(os.path.join(path, TIMESTEPS_USED_FILENAME)) as f:
        timesteps = yaml.safe_load(f)
    with fsspec.open(os.path.join(path, TRAINING_FIG_FILENAME), "wb") as f:
        gallery.plot_daily_and_hourly_hist(timesteps).savefig(f, dpi=90)
    return {"Time distribution of training samples": [TRAINING_FIG_FILENAME]}


def _write_report(output_dir, sections, metadata, title):
    filename = title.replace(" ", "_") + ".html"
    html_report = report.create_html(sections, title, metadata=metadata)
    with fsspec.open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


def _url_to_datetime(url):
    return vcm.cast_to_datetime(
        vcm.parse_datetime_from_str(vcm.parse_timestep_str_from_path(url))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=str, help="Location of training data")
    parser.add_argument(
        "train_config_file", type=str, help="Path for training configuration yaml file"
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
    args = parser.parse_args()
    args.train_data_path = os.path.join(args.train_data_path, "train")
    train_config = load_model_training_config(
        args.train_config_file, args.train_data_path
    )
    batched_data = load_data_generator(train_config)

    model, training_urls_used = train_model(batched_data, train_config)
    save_model(args.output_data_path, model, MODEL_FILENAME)
    timesteps_used = list(map(_url_to_datetime, training_urls_used))
    _save_config_output(args.output_data_path, train_config, timesteps_used)
    report_sections = _create_report_plots(args.output_data_path)
    report_metadata = {**vars(args), **vars(train_config)}
    _write_report(args.output_data_path, report_sections, report_metadata, REPORT_TITLE)

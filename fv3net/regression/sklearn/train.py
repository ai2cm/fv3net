import argparse
import fsspec
import joblib
import logging
import os
import yaml

from datetime import datetime
from dataclasses import dataclass
from typing import List

import report
import gallery
from fv3net.regression.dataset_handler import BatchGenerator
from fv3net.regression.sklearn.wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
import vcm.cloud.fsspec
import vcm

MODEL_CONFIG_FILENAME = "training_config.yml"
MODEL_FILENAME = "sklearn_model.pkl"
TIMESTEPS_USED_FILENAME = "timesteps_used.yml"
REPORT_TITLE = "ML model training report"
TRAINING_FIG_FILENAME = "count_of_training_times_used.png"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("ml_training.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


@dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info

    """

    model_type: str
    gcs_data_dir: str
    hyperparameters: dict
    num_batches: int
    files_per_batch: int
    input_variables: List[str]
    output_variables: List[str]
    gcs_project: str = "vcm-ml"
    random_seed: int = 1234
    mask_to_surface_type: str = "none"
    coord_z_center: str = "z"
    init_time_dim: str = "initial_time"

    def validate_number_train_batches(self, batch_generator):
        """ Since number of training files specified may be larger than
        the actual number available, this adds an attribute num_batches_used
        that keeps information about the actual number of training batches
        used.

        Args:
            batch_generator (BatchGenerator)
        """
        self.num_batches_used = batch_generator.num_batches


def load_model_training_config(config_path, gcs_data_dir):
    """

    Args:
        config_path: location of .yaml that contains config for model training

    Returns:
        ModelTrainingConfig object
    """
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    config_dict["gcs_data_dir"] = gcs_data_dir
    config = ModelTrainingConfig(**config_dict)
    return config


def load_data_generator(train_config):
    """

    Args:
        train_config: ModelTrainingConfig object

    Returns:
        iterator that generates xr datasets for training batches
    """
    data_vars = train_config.input_variables + train_config.output_variables
    ds_batches = BatchGenerator(
        data_vars,
        train_config.gcs_data_dir,
        train_config.files_per_batch,
        train_config.num_batches,
        random_seed=train_config.random_seed,
        mask_to_surface_type=train_config.mask_to_surface_type,
    )
    return ds_batches


def _get_regressor(train_config):
    """Reads the model type from the model training config and initializes a regressor
    with hyperparameters from config.

    Args:
        train_config: ModelTrainingConfig object

    Returns:
        regressor (varies depending on model type)
    """
    model_type = train_config.model_type.replace(" ", "").replace("_", "")
    if "rf" in model_type or "randomforest" in model_type:
        from sklearn.ensemble import RandomForestRegressor

        regressor = RandomForestRegressor(**train_config.hyperparameters, n_jobs=-1)
    else:
        raise ValueError(
            f"Model type {train_config.model_type} not implemented. "
            "Options are "
            " 1) random forest (contains keywords 'rf' "
            "or 'random forest') "
        )
    return regressor


def train_model(batched_data, train_config):
    """

    Args:
        batched_data: iterator that yields training batch datasets
        train_config: ModelTrainingConfig object
        targets_for_normalization: array of sample output data used to save norm and std
            dev to the StandardScaler transformer

    Returns:
        trained sklearn model wrapper object
    """
    base_regressor = _get_regressor(train_config)
    target_transformer = StandardScaler()
    transform_regressor = TransformedTargetRegressor(base_regressor, target_transformer)
    batch_regressor = RegressorEnsemble(transform_regressor)
    model_wrapper = SklearnWrapper(batch_regressor)

    train_config.validate_number_train_batches(batched_data)
    training_urls_to_attempt = batched_data.train_file_batches
    training_urls_used = []

    for i, batch in enumerate(
        batched_data.generate_batches(
            train_config.coord_z_center, train_config.init_time_dim
        )
    ):
        logger.info(f"Fitting batch {i}/{batched_data.num_batches}")
        try:
            model_wrapper.fit(
                input_vars=train_config.input_variables,
                output_vars=train_config.output_variables,
                sample_dim="sample",
                data=batch,
            )
            logger.info(f"Batch {i} done fitting.")
            training_urls_used += training_urls_to_attempt[i]
        except ValueError as e:
            logger.error(f"Error training on batch {i}: {e}")
            train_config.num_batches_used -= 1
            continue

    return model_wrapper, training_urls_used


def save_output(output_url, model, config, timesteps):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    model_url = os.path.join(output_url, MODEL_FILENAME)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)
    timesteps_url = os.path.join(output_url, TIMESTEPS_USED_FILENAME)

    with fs.open(model_url, "wb") as f:
        joblib.dump(model, f)

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


def _write_training_html_report(path, sections, metadata):
    """Write html report to path, given sections and metadata"""
    html_report = report.create_html(sections, REPORT_TITLE, metadata=metadata)
    report_filename = REPORT_TITLE.replace(" ", "_") + ".html"
    with fsspec.open(os.path.join(path, report_filename), "w") as f:
        f.write(html_report)


def _url_to_datetime(url):
    dt = vcm.parse_datetime_from_str(vcm.parse_timestep_str_from_path(url))
    # ensure returning a python datetime (i.e. not cftime.DatetimeJulian)
    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


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
    timesteps_used = list(map(_url_to_datetime, training_urls_used))
    save_output(args.output_data_path, model, train_config, timesteps_used)
    report_sections = _create_report_plots(args.output_data_path)
    report_metadata = {**vars(args), **vars(train_config)}
    _write_training_html_report(args.output_data_path, report_sections, report_metadata)

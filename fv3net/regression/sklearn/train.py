import argparse
import joblib
import os
import yaml

from dataclasses import dataclass
from datetime import datetime
from shutil import copyfile, rmtree
from typing import List

from fv3net.regression.dataset_handler import BatchGenerator
from fv3net.regression.sklearn.wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from vcm.cloud import gsutil

MODEL_CONFIG_FILENAME = "training_config.yml"
MODEL_FILENAME = "sklearn_model.pkl"
OUTPUT_DIR_SUFFIX = "model_training_files"


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


def load_model_training_config(config_path):
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
    for i, batch in enumerate(batched_data.generate_batches("train")):
        print(f"Fitting batch {i}/{batched_data.num_train_batches}")
        model_wrapper.fit(
            input_vars=train_config.input_variables,
            output_vars=train_config.output_variables,
            sample_dim="sample",
            data=batch,
        )
        print(f"Batch {i} done fitting.")
    return model_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config-file",
        type=str,
        required=True,
        help="Path for training configuration yaml file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="''",
        help="Optional- provide a directory in which model and config will be saved.",
    )
    parser.add_argument(
        "--remote-output-url",
        type=str,
        required=False,
        help="Optional remote location to save config and trained model.",
    )
    parser.add_argument(
        "--delete-local-results-after-upload",
        type=bool,
        default=True,
        help="If results are uploaded to remote storage, "
        "remove local copy after upload.",
    )
    args = parser.parse_args()
    train_config = load_model_training_config(args.train_config_file)
    batched_data = load_data_generator(train_config)

    model = train_model(batched_data, train_config)

    # model and config are saved with timestamp prefix so that they can be
    # matched together
    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    output_dir = f"{timestamp}_{OUTPUT_DIR_SUFFIX}"
    copyfile(
        args.train_config_file,
        os.path.join(output_dir, f"{timestamp}_{MODEL_CONFIG_FILENAME}"),
    )
    joblib.dump(model, os.path.join(output_dir, f"{timestamp}_{MODEL_FILENAME}"))

    if args.remote_output_url:
        gsutil.copy(output_dir, args.remote_output_url)
        if args.delete_local_results_after_upload is True:
            rmtree(output_dir)

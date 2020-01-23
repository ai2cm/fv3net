import argparse
import joblib
import numpy as np
import os
import yaml

from dataclasses import dataclass
from datetime import datetime
from importlib.resources import path
from shutil import copyfile
from typing import List

from fv3net.regression.dataset_handler import BatchGenerator
from fv3net.regression.sklearn.wrapper import SklearnWrapper, BatchTransformRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


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
    batch_regressor = BatchTransformRegressor(transform_regressor)

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
        "--model-output-filename",
        type=str,
        required=True,
        help="Path for writing trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="''",
        help="Optional- provide a directory in which model and config will be saved.",
    )
    args = parser.parse_args()
    train_config = load_model_training_config(args.train_config_file)
    batched_data = load_data_generator(train_config)

    model = train_model(batched_data, train_config)

    # model and config are saved with timestamp prefix so that they can be
    # matched together
    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    copyfile(
        args.train_config_file,
        os.path.join(args.output_dir, f"{timestamp}_training_config.yml"),
    )
    joblib.dump(model, f"{timestamp}_{args.model_output_filename}")

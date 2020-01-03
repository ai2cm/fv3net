import argparse
from dataclasses import dataclass
from importlib.resources import path
from typing import List

import joblib
import numpy as np
import yaml

from fv3net.regression.dataset_handler import BatchGenerator
from fv3net.regression.sklearn.wrapper import (
    SklearnWrapper,
    BatchTransformRegressor,
)
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
    batch_size: int
    train_frac: float
    test_frac: float
    input_variables: List[str]
    output_variables: List[str]
    gcs_project: str = "vcm-ml"


def get_outputs_for_normalization(output_normalization_file):
    """

    Args:
        output_normalization_file: must be .npy, .txt., or .dat

    Returns:
        np array of sample target values for use in StandardScaler
    """
    if output_normalization_file == "default":
        with path("fv3net.regression.sklearn", "default_norm_outputs.dat") as f:
            sample_outputs = np.loadtxt(f)
    else:
        with open(output_normalization_file, "r") as f:
            if output_normalization_file.split(".")[-1] == ".npy":
                sample_outputs = np.load(f)
            elif output_normalization_file.split(".")[-1] in [".txt", ".dat"]:
                sample_outputs = np.loatxt(f)
            else:
                raise ValueError(
                    "Provide either a .npy array file, or '.txt' or '.dat'"
                    "with one column per output feature"
                )

    return sample_outputs


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
    ds_batches = BatchGenerator(
        train_config.gcs_data_dir,
        train_config.batch_size,
        train_config.train_frac,
        train_config.test_frac,
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
    elif "gbt" in model_type or "boostedtrees" in model_type:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        regressor = MultiOutputRegressor(
            XGBRegressor(**train_config.hyperparameters), n_jobs=-1
        )
    else:
        raise ValueError(
            f"Model type {train_config.model_type} not implemented. "
            "Options are random forest (contains keywords 'rf' "
            "or 'random forest') or gradient boosted trees "
            "(contains keywords 'gbt' or 'boosted trees')."
        )
    return regressor


def train_model(batched_data, train_config, targets_for_normalization):
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
    target_transformer.fit(targets_for_normalization)
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
    return model_wrapper.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config-file",
        type=str,
        required=True,
        help="Path for training configuration yaml file",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        required=True,
        help="Path for writing trained model",
    )
    parser.add_argument(
        "--target-normalization-file",
        type=str,
        default="default",
        help="File that contains array of sample output for use in StandardScaler",
    )
    args = parser.parse_args()
    train_config = load_model_training_config(args.train_config_file)
    batched_data = load_data_generator(train_config)
    targets_for_normalization = get_outputs_for_normalization(
        args.target_normalization_file
    )
    model = train_model(batched_data, train_config, targets_for_normalization)
    joblib.dump(model, args.model_output_path)

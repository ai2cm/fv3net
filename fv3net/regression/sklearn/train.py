import fsspec
import joblib
import logging
import os
import yaml
import xarray as xr
from dataclasses import dataclass
from typing import Iterable, Sequence

from .. import loaders
from .wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__file__)


@dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    model_type: str
    hyperparameters: dict
    input_variables: Iterable[str]
    output_variables: Iterable[str]
    batch_loader: dict


def load_model_training_config(config_path: str) -> ModelTrainingConfig:
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
    return ModelTrainingConfig(**config_dict)


def load_data_sequence(
    data_path: str, train_config: ModelTrainingConfig
) -> Sequence[xr.Dataset]:
    """
    Args:
        data_path: data location
        train_config: model training configuration

    Returns:
        FunctionOutputSequence: wrapper object that is effectively a
        Sequence[xr.Dataset] when iterated over in training
    """
    batch_function_name = train_config.batch_loader["batch_function"]
    batch_function = getattr(loaders, batch_function_name)
    ds_batches = batch_function(
        data_path,
        list(train_config.input_variables) + list(train_config.output_variables),
        **train_config.batch_loader["batch_kwargs"],
    )
    return ds_batches


def _get_regressor(train_config: ModelTrainingConfig):
    """Reads the model type from the model training config and initializes a regressor
    with hyperparameters from config.

    Args:
        train_config: model training configuration

    Returns:
        regressor (varies depending on model type)
    """
    model_type = train_config.model_type.replace(" ", "").replace("_", "")
    if "rf" in model_type or "randomforest" in model_type:
        train_config.hyperparameters["random_state"] = train_config.hyperparameters.get(
            "random_state", 0
        )
        train_config.hyperparameters["n_jobs"] = train_config.hyperparameters.get(
            "n_jobs", -1
        )
        regressor = RandomForestRegressor(**train_config.hyperparameters)
    else:
        raise ValueError(
            f"Model type {train_config.model_type} not implemented. "
            "Options are "
            " 1) random forest (contains keywords 'rf' "
            "or 'random forest') "
        )
    return regressor


def _get_transformed_batch_regressor(train_config):
    base_regressor = _get_regressor(train_config)
    target_transformer = StandardScaler()
    transform_regressor = TransformedTargetRegressor(base_regressor, target_transformer)
    batch_regressor = RegressorEnsemble(transform_regressor)
    model_wrapper = SklearnWrapper(batch_regressor)
    return model_wrapper


def train_model(batched_data: Sequence[xr.Dataset], train_config: ModelTrainingConfig):
    """
    Args:
        batched_data: training batch datasets
        train_config: model training configuration
        targets_for_normalization: array of sample output data used to save norm and std
            dev to the StandardScaler transformer

    Returns:
        trained sklearn model wrapper object
    """

    model_wrapper = _get_transformed_batch_regressor(train_config)
    for i, batch in enumerate(batched_data):
        logger.info(f"Fitting batch {i}/{len(batched_data)}")
        model_wrapper.fit(
            input_vars=train_config.input_variables,
            output_vars=train_config.output_variables,
            sample_dim="sample",
            data=batch,
        )
        logger.info(f"Batch {i} done fitting.")

    return model_wrapper


def save_model(output_url: str, model, model_filename: str):
    """Save model to {output_url}/{model_filename} using joblib.dump"""
    fs, _, _ = fsspec.get_fs_token_paths(output_url)
    fs.makedirs(output_url, exist_ok=True)
    model_url = os.path.join(output_url, model_filename)
    with fs.open(model_url, "wb") as f:
        joblib.dump(model, f)

from copy import copy
from typing import List, Union, Sequence, Tuple
import xarray as xr

from loaders import batches, Map
from .config import ModelTrainingConfig


def load_data_sequence(
    data_path: Union[List, tuple, str], train_config: ModelTrainingConfig
) -> Map[xr.Dataset]:
    """
    Args:
        data_path: data location
        train_config: model training configuration

    Returns:
        Sequence of datasets iterated over in training
    """
    batch_function = getattr(batches, train_config.batch_function)
    ds_batches = batch_function(
        data_path,
        list(train_config.input_variables)
        + list(train_config.output_variables)
        + list(train_config.additional_variables),  # type: ignore
        **train_config.batch_kwargs,
    )
    return ds_batches


def check_validation_train_overlap(
    train: Sequence[str], validate: Sequence[str]
) -> None:
    overlap = set(train) & set(validate)
    if overlap:
        raise ValueError(
            f"Timestep(s) {overlap} are in both train and validation sets."
        )


def validation_timesteps_config(train_config: ModelTrainingConfig):
    val_config = copy(train_config)
    val_config.batch_kwargs["timesteps"] = train_config.validation_timesteps
    val_config.batch_kwargs["timesteps_per_batch"] = len(
        train_config.validation_timesteps
    )
    return val_config

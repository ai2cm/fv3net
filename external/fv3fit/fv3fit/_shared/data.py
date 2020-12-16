from typing import List, Union, Sequence, Tuple
import xarray as xr

from loaders import batches, Map
from .config import ModelTrainingConfig


def train_validation_split_batches(
    timesteps: Sequence,
    timesteps_per_training_batch: int,
    timesteps_per_validation_batch: int = 1,
) -> Tuple[Sequence, Sequence]:
    n_batches = len(timesteps) // (
        timesteps_per_training_batch + timesteps_per_validation_batch
    )
    n_train = n_batches * timesteps_per_training_batch
    sorted_timesteps = sorted(timesteps)
    leftover = len(timesteps) % (
        timesteps_per_training_batch + timesteps_per_validation_batch
    )
    if leftover > 1:
        leftover_train = min(leftover - 1, timesteps_per_training_batch)
        n_train += leftover_train
    return (sorted_timesteps[:n_train], sorted_timesteps[n_train:])


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


def load_validation_data_sequence(
    data_path: Union[List, tuple, str], train_config: ModelTrainingConfig,
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

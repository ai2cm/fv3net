from copy import copy
from typing import Sequence, Optional
import xarray as xr

from .._shared import ModelTrainingConfig, load_data_sequence


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
        train_config.validation_timesteps  # type: ignore
    )
    return val_config


def validation_dataset(
    data_path,
    train_config: ModelTrainingConfig,
) -> Optional[xr.Dataset]:
    if len(train_config.validation_timesteps) > 0:
        check_validation_train_overlap(
            train_config.batch_kwargs["timesteps"], train_config.validation_timesteps,
        )
        validation_config = validation_timesteps_config(train_config)
        # validation config puts all data in one batch
        validation_dataset_sequence = load_data_sequence(
            data_path, validation_config
        )
        if len(validation_dataset_sequence) > 1:
            raise ValueError(
                "Something went wrong! "
                "All validation data should be concatenated into a single batch. There "
                f"are {len(validation_dataset_sequence)} batches in the sequence."
            )
        return validation_dataset_sequence[0]
    else:
        validation_dataset = None
    return validation_dataset

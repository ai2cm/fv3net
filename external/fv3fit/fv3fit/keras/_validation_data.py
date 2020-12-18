from copy import copy
from typing import Sequence
from .._shared import ModelTrainingConfig


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

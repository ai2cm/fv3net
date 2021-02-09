from typing import Sequence
from synth import (  # noqa: F401
    dataset_fixtures_dir,
    data_source_name,
    nudging_dataset_path,
    fine_res_dataset_path,
    data_source_path,
    grid_dataset,
)
import xarray as xr
from fv3fit._shared import load_data_sequence
from fv3fit._shared.config import ModelTrainingConfig
import pytest


batch_kwargs = {
    "needs_grid": False,
    "res": "c8_random_values",
    "timesteps_per_batch": 1,
    "mapping_function": "open_merged_nudged",
    "timesteps": ["20160801.001500"],
    "mapping_kwargs": {
        "i_start": 0,
        "rename_vars": {
            "air_temperature_tendency_due_to_nudging": "dQ1",
            "specific_humidity_tendency_due_to_nudging": "dQ2",
        },
    },
}

@pytest.fixture
def train_config() -> ModelTrainingConfig:
    return ModelTrainingConfig(
        model_type="DenseModel",
        hyperparameters={"width": 3, "depth": 2},
        input_variables=["air_temperature", "specific_humidity"],
        output_variables= ["dQ1", "dQ2"],
        batch_function="batches_from_geodata",
        batch_kwargs=batch_kwargs,
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=None,
        random_seed=0,
        validation_timesteps=None,
        data_path=data_source_path,
    )


@pytest.fixture
def training_batches(
    data_source_name: str,  # noqa: F811
    data_source_path: str,  # noqa: F811
    train_config: ModelTrainingConfig,
) -> Sequence[xr.Dataset]:
    batched_data = load_data_sequence(data_source_path, train_config)
    return batched_data

from typing import Iterable, Sequence, Optional
from synth import (  # noqa: F401
    dataset_fixtures_dir,
    data_source_name,
    nudging_dataset_path,
    fine_res_dataset_path,
    data_source_path,
    grid_dataset,
)
import xarray as xr
from fv3fit._shared import ModelTrainingConfig, load_data_sequence
import pytest
import tempfile
import yaml


@pytest.fixture(params=[None])
def validation_timesteps(request) -> Optional[Sequence[str]]:
    return request.param


@pytest.fixture
def input_variables() -> Iterable[str]:
    return ["air_temperature", "specific_humidity"]


@pytest.fixture
def output_variables() -> Iterable[str]:
    return ["dQ1", "dQ2"]


def get_batch_kwargs(data_source_name: str) -> dict:  # noqa: F811
    if data_source_name == "nudging_tendencies":
        return {
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
    elif data_source_name == "fine_res_apparent_sources":
        return {
            "needs_grid": False,
            "res": "c8_random_values",
            "timesteps_per_batch": 1,
            "mapping_function": "open_fine_res_apparent_sources",
            "timesteps": ["20160801.001500"],
            "mapping_kwargs": {
                "rename_vars": {
                    "delp": "pressure_thickness_of_atmospheric_layer",
                    "grid_xt": "x",
                    "grid_yt": "y",
                    "pfull": "z",
                }
            },
        }


@pytest.fixture
def train_config(
    model_type: str,
    hyperparameters: dict,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    data_source_name,
    validation_timesteps: Optional[Sequence[str]],
) -> ModelTrainingConfig:
    return ModelTrainingConfig(
        data_path="train_data_path",
        model_type=model_type,
        hyperparameters=hyperparameters,
        input_variables=input_variables,
        output_variables=output_variables,
        batch_function="batches_from_geodata",
        batch_kwargs=get_batch_kwargs(data_source_name),
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=None,
        random_seed=0,
        validation_timesteps=validation_timesteps,
    )


@pytest.fixture
def training_batches(
    data_source_path: str, train_config: ModelTrainingConfig,  # noqa: F811
) -> Sequence[xr.Dataset]:
    batched_data = load_data_sequence(data_source_path, train_config)
    return batched_data


@pytest.fixture
def data_info(data_source_path, data_source_name):
    return dict(
        data_path=data_source_path, batch_kwargs=get_batch_kwargs(data_source_name)
    )

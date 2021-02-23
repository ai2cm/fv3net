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


@pytest.fixture()
def batch_function(model_type: str) -> str:
    return "batches_from_geodata"


@pytest.fixture()
def batch_kwargs(data_source_name: str) -> dict:  # noqa: F811
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
    batch_function: str,
    batch_kwargs: dict,
    validation_timesteps: Optional[Sequence[str]],
) -> ModelTrainingConfig:
    return ModelTrainingConfig(
        data_path="train_data_path",
        model_type=model_type,
        hyperparameters=hyperparameters,
        input_variables=input_variables,
        output_variables=output_variables,
        batch_function=batch_function,
        batch_kwargs=batch_kwargs,
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=None,
        random_seed=0,
        validation_timesteps=validation_timesteps,
    )


@pytest.fixture
def train_config_filename(
    model_type: str,
    hyperparameters: dict,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    batch_function: str,
    batch_kwargs: dict,
    validation_timesteps: Optional[Sequence[str]],
) -> str:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.dump(
            {
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "input_variables": input_variables,
                "output_variables": output_variables,
                "batch_function": batch_function,
                "batch_kwargs": batch_kwargs,
                "scaler_type": "standard",
                "scaler_kwargs": {},
                "additional_variables": None,
                "random_seed": 0,
                "validation_timesteps": validation_timesteps,
            },
            f,
        )
        yield f.name


@pytest.fixture
def training_batches(
    data_source_name: str,  # noqa: F811
    data_source_path: str,  # noqa: F811
    train_config: ModelTrainingConfig,
) -> Sequence[xr.Dataset]:
    batched_data = load_data_sequence(data_source_path, train_config)
    return batched_data

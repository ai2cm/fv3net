from typing import Iterable, Sequence, Optional
import xarray as xr
from fv3fit._shared import ModelTrainingConfig, load_data_sequence
import pytest

import numpy as np
import xarray
import cftime


@pytest.fixture(params=[None])
def validation_timesteps(request) -> Optional[Sequence[str]]:
    return request.param


@pytest.fixture
def input_variables() -> Iterable[str]:
    return [
        "air_temperature",
        "specific_humidity",
        "pressure_thickness_of_atmospheric_layer",
    ]


@pytest.fixture
def output_variables() -> Iterable[str]:
    return ["dQ1", "dQ2", "total_precipitation_rate"]


@pytest.fixture
def data_info(tmpdir):

    x, y, z, tile, time = (8, 8, 79, 6, 2)
    arr = np.zeros((time, tile, z, y, x))
    arr_surf = np.zeros((time, tile, y, x))
    dims = ["time", "tile", "z", "y", "x"]
    dims_surf = ["time", "tile", "y", "x"]

    data = xarray.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "downward_shortwave": (dims_surf, arr_surf),
            "net_shortwave": (dims_surf, arr_surf),
            "downward_longwave": (dims_surf, arr_surf),
            "total_precipitation_rate": (dims_surf, arr_surf),
            "pressure_thickness_of_atmospheric_layer": (dims_surf, arr_surf),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        },
        coords={
            "time": [
                cftime.DatetimeJulian(2016, 8, 1),
                cftime.DatetimeJulian(2016, 8, 2),
            ]
        },
    )

    data.to_zarr(str(tmpdir), consolidated=True)
    return dict(
        data_path=str(tmpdir),
        batch_kwargs=dict(
            mapping_function="open_zarr",
            timesteps=["20160801.000000"],
            needs_grid=False,
            res="c8_random_values",
            timesteps_per_batch=1,
        ),
        validation_timesteps=["20160802.000000"],
    )


@pytest.fixture
def train_config(
    model_type: str,
    hyperparameters: dict,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    data_info,
    validation_timesteps: Optional[Sequence[str]],
) -> ModelTrainingConfig:
    return ModelTrainingConfig(
        data_path="train_data_path",
        model_type=model_type,
        hyperparameters=hyperparameters,
        input_variables=input_variables,
        output_variables=output_variables,
        batch_function="batches_from_geodata",
        batch_kwargs=data_info["batch_kwargs"],
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=None,
        random_seed=0,
        validation_timesteps=validation_timesteps,
    )


@pytest.fixture
def training_batches(
    data_info: str, train_config: ModelTrainingConfig,  # noqa: F811
) -> Sequence[xr.Dataset]:
    batched_data = load_data_sequence(data_info["data_path"], train_config)
    return batched_data

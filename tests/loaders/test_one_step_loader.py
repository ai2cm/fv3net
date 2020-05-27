import atexit
import os
import numpy as np
import pytest
import synth
import tempfile
import xarray as xr

from fv3net.regression.loaders._one_step import (
    _load_one_step_batch,
    _load_datasets,
    load_one_step_batches,
    TimestepMapper,
)

ONE_STEP_ZARR_SCHEMA = "one_step_zarr_schema.json"
Z_DIM_SIZE = 79
TIMESTEP_LIST = [f"2020050{i}.000000" for i in range(4)]
DATA_VARS = ["air_temperature", "specific_humidity"]


@pytest.fixture
def test_timestep_mapper(datadir):
    with open(os.path.join(datadir, ONE_STEP_ZARR_SCHEMA)) as f:
        schema = synth.load(f)
    one_step_dataset = synth.generate(schema)
    for timestep in TIMESTEP_LIST:
        output_path = os.path.join(datadir, f"{timestep}.zarr")
        one_step_dataset.to_zarr(output_path, consolidated=True)    
    timestep_mapper = TimestepMapper(str(datadir))
    return timestep_mapper


def test_load_datasets(test_timestep_mapper):
    ds_list = _load_datasets(
        test_timestep_mapper, TIMESTEP_LIST[:2]
    )
    assert len(ds_list) == 2


def test__load_one_step_batch(test_timestep_mapper):
    ds = _load_one_step_batch(
        timestep_mapper=test_timestep_mapper,
        data_vars=["air_temperature", "specific_humidity"],
        rename_variables={},
        init_time_dim_name="time",
        timestep_list=TIMESTEP_LIST[:3],
    )
    assert len(ds["time"]) == 3


def test_load_one_step_batches(datadir, test_timestep_mapper):
    batched_data_sequence = load_one_step_batches(
        str(datadir),
        DATA_VARS,
        files_per_batch=2,
        num_batches=2,
    )
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)

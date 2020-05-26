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
    _shuffled, 
    _chunk_indices
)

ONE_STEP_ZARR_SCHEMA = "tests/loaders/one_step_zarr_schema.json"
Z_DIM_SIZE = 79
TEMP_DIR = tempfile.TemporaryDirectory()


def _cleanup_temp_dir(temp_dir):
    print(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


def _write_synth_dataset(output_path):
    with open(ONE_STEP_ZARR_SCHEMA) as f:
        schema = synth.load(f)
    one_step_dataset = synth.generate(schema)
    one_step_dataset.to_zarr(output_path, consolidated=True)


atexit.register(_cleanup_temp_dir, TEMP_DIR)
for i in range(4):
    output_path = os.path.join(TEMP_DIR.name, f"2020050{i}.000000.zarr")
    _write_synth_dataset(output_path)


@pytest.fixture
def test_timestep_mapper():
    timestep_mapper = TimestepMapper(TEMP_DIR.name)
    return timestep_mapper


def test_load_datasets(test_timestep_mapper):
    ds_list = _load_datasets(
        test_timestep_mapper, ["20200500.000000", "20200501.000000"]
    )
    assert len(ds_list) == 2


def test__load_one_step_batch(test_timestep_mapper):
    ds = _load_one_step_batch(
        timestep_mapper=test_timestep_mapper,
        data_vars=["air_temperature", "specific_humidity"],
        rename_variables={},
        init_time_dim_name="time",
        timestep_list=["20200500.000000", "20200501.000000", "20200502.000000"],
    )
    assert len(ds["time"]) == 3


def test_load_one_step_batches():
    batched_data_sequence = load_one_step_batches(
        TEMP_DIR.name,
        ["air_temperature", "specific_humidity"],
        files_per_batch=2,
        num_batches=2,
    )
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE


# The tests below are for the stack/shuffle transform that will
# be refactored to common usage for all data sources.
# Currently they remain in _one_step.py

def test__chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _chunk_indices(chunks)
    assert ans == expected


def _dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test__shuffled():
    dataset = _dataset("sample")
    dataset.isel(sample=1)
    _shuffled(dataset, "sample", np.random.RandomState(1))


def test__shuffled_dask():
    dataset = _dataset("sample").chunk()
    _shuffled(dataset, "sample", np.random.RandomState(1))

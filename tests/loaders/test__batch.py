import os
import pytest
import synth
import xarray as xr
import numpy as np

from fv3net.regression.loaders._batch import (
    _mapper_to_batches,
    _mapper_to_diagnostic_sequence,
    _load_batch,
)

DATA_VARS = ["air_temperature", "specific_humidity"]
Z_DIM_SIZE = 79


class MockDatasetMapper:
    def __init__(self, schema: synth.DatasetSchema):
        self._schema = schema
        self._keys = [f"2020050{i}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = synth.generate(self._schema)
        ds.coords["initial_time"] = [key]
        return ds

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


@pytest.fixture
def mapper(datadir):
    one_step_zarr_schema = "one_step_zarr_schema.json"
    # uses the one step schema but final mapper
    # functions the same for all data sources
    with open(os.path.join(datadir, one_step_zarr_schema)) as f:
        schema = synth.load(f)
    mapper = MockDatasetMapper(schema)
    return mapper


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test__load_batch(mapper):
    ds = _load_batch(
        mapper=mapper,
        data_vars=["air_temperature", "specific_humidity"],
        rename_variables={},
        init_time_dim_name="time",
        keys=mapper.keys(),
    )
    assert len(ds["time"]) == 4


def test__mapper_to_batches(mapper):
    batched_data_sequence = _mapper_to_batches(mapper, DATA_VARS, timesteps_per_batch=2)
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


@pytest.mark.parametrize(
    "total_times,times_per_batch,valid_num_batches", [(3, 1, 3), (3, 2, 1)]
)
def test__mapper_to_batches_timestep_list(
    mapper, total_times, times_per_batch, valid_num_batches
):
    timestep_list = list(mapper.keys())[:total_times]
    batched_data_sequence = _mapper_to_batches(
        mapper, DATA_VARS, timesteps_per_batch=times_per_batch, timesteps=timestep_list
    )
    print(batched_data_sequence._args)
    assert len(batched_data_sequence) == valid_num_batches
    timesteps_used = sum(batched_data_sequence._args, ())  # flattens list
    assert set(timesteps_used).issubset(timestep_list)


def test__mapper_to_batches_invalid_times(mapper):
    invalid_times = list(mapper.keys())[:2] + ["20000101.000000", "20000102.000000"]
    with pytest.raises(ValueError):
        _mapper_to_batches(
            mapper, DATA_VARS, timesteps_per_batch=2, timesteps=invalid_times
        )


def test__mapper_to_diagnostic_sequence(mapper):
    batched_data_sequence = _mapper_to_diagnostic_sequence(
        mapper, DATA_VARS, timesteps_per_batch=2,
    )
    assert len(batched_data_sequence) == len(mapper) // 2 + len(mapper) % 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)

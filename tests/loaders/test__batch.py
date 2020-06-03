import numpy as np
import os
import pytest
import synth
import xarray as xr

from fv3net.regression.loaders._batch import (
    mapper_to_batches,
    _load_batch,
    BatchSequence,
    mapper_to_diagnostic_sequence,
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


@pytest.fixture
def batch_sequence(mapper, random_state):
    return BatchSequence(
        mapper, files_per_batch=2, num_batches=2, random_state=random_state
    )


def test__load_batch(batch_sequence):
    ds = _load_batch(
        batch_sequence=batch_sequence,
        data_vars=["air_temperature", "specific_humidity"],
        rename_variables={},
        init_time_dim_name="time",
        batch_index=0,
    )
    assert len(ds["time"]) == 2


def test_mapper_to_batches(mapper):
    batched_data_sequence = mapper_to_batches(
        mapper, DATA_VARS, timesteps_per_batch=2, num_batches=2
    )
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


def test_mapper_to_diagnostic_sequence(mapper):
    batched_data_sequence = mapper_to_diagnostic_sequence(
        mapper, DATA_VARS, timesteps_per_batch=2, num_batches=2
    )
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


@pytest.mark.parametrize(
    "total_times,times_per_batch,num_batches,valid_num_batches",
    [
        (5, 1, None, 5),
        (5, 2, None, 2),
        (5, 2, 1, 1),
        (2, 6, None, None),
        (2, 1, 3, None),
        (0, 5, None, None),
        (5, 0, None, None),
    ],
)
def test__validated_num_batches(
    total_times, times_per_batch, num_batches, valid_num_batches
):
    if valid_num_batches:
        assert (
            BatchSequence._validated_num_batches(
                total_times, times_per_batch, num_batches
            )
            == valid_num_batches
        )
    else:
        with pytest.raises(ValueError):
            BatchSequence._validated_num_batches(
                total_times, times_per_batch, num_batches,
            )


class MockDatasetMapperNoData:
    def __init__(self, n_timesteps):
        self._keys = [f"2020050{i}.000000" for i in range(n_timesteps)]

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


def mapper_timesteps(timesteps):
    return MockDatasetMapperNoData(timesteps)


@pytest.mark.parametrize(
    "timesteps_per_batch, num_batches, total_num_timesteps, valid",
    (
        [3, 4, 12, True],
        [4, 2, 12, True],
        [1, 1, 12, True],
        [1, 1, 1, True],
        [4, 2, 0, False],
        [0, 2, 12, False],
        [3, 0, 12, False],
        [3, 5, 12, False],
        [3, 0, 12, False],
    ),
)
def test__select_batch_timesteps(
    timesteps_per_batch, num_batches, total_num_timesteps, valid
):
    mapper = mapper_timesteps(total_num_timesteps)
    random_state = np.random.RandomState(0)
    timesteps = mapper.keys()
    if valid:
        batch_sequence = BatchSequence(
            mapper, timesteps_per_batch, num_batches, random_state
        )
        batched_times = batch_sequence.batches
        timesteps_seen = []
        assert len(batched_times) == num_batches
        for batch in batched_times:
            assert len(np.unique(batch)) == timesteps_per_batch
            assert set(batch).isdisjoint(timesteps_seen) and set(batch).issubset(
                timesteps
            )
            timesteps_seen += batch
    else:
        with pytest.raises(Exception):
            batch_sequence = BatchSequence(
                mapper, timesteps_per_batch, num_batches, random_state
            )

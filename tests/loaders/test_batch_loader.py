import os
import pytest
import synth
import xarray as xr

from fv3net.regression.loaders.batch import (
    load_batches,
    _load_batch,
    _load_datasets,
    _select_batch_timesteps,
    _validated_num_batches,
)

ONE_STEP_ZARR_SCHEMA = "one_step_zarr_schema.json"
TIMESTEP_LIST = [f"2020050{i}.000000" for i in range(4)]
DATA_VARS = ["air_temperature", "specific_humidity"]
Z_DIM_SIZE = 79


class MockDatasetMapper:
    def generate_mapping(self, dir):
        # uses the one step schema but final mapper
        # functions the same for all data sources
        with open(os.path.join(dir, ONE_STEP_ZARR_SCHEMA)) as f:
            schema = synth.load(f)
        dataset = synth.generate(schema)
        self.mapping = {}
        for i in range(4):
            self.mapping[f"2020050{i}.000000"] = dataset

    def __getitem__(self, key: str) -> xr.Dataset:
        return self.mapping[key]

    def keys(self):
        return list(self.mapping.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


@pytest.fixture
def test_mapper(datadir):
    mapper = MockDatasetMapper()
    mapper.generate_mapping(datadir)
    return mapper


def test__load_batch(test_mapper):
    ds = _load_batch(
        timestep_mapper=test_mapper,
        data_vars=["air_temperature", "specific_humidity"],
        rename_variables={},
        init_time_dim_name="time",
        timestep_list=TIMESTEP_LIST[:3],
    )
    assert len(ds["time"]) == 3


def test__load_datasets(test_mapper):
    ds_list = _load_datasets(test_mapper, TIMESTEP_LIST[:2])
    assert len(ds_list) == 2


def test_load_batches(test_mapper):
    batched_data_sequence = load_batches(
        test_mapper, DATA_VARS, files_per_batch=2, num_batches=2
    )
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


def test__validated_num_batches():
    assert (
        _validated_num_batches(
            total_num_input_files=5, files_per_batch=1, num_batches=None
        )
        == 5
    )
    assert (
        _validated_num_batches(
            total_num_input_files=5, files_per_batch=2, num_batches=None
        )
        == 2
    )
    assert (
        _validated_num_batches(
            total_num_input_files=5, files_per_batch=2, num_batches=1
        )
        == 1
    )
    with pytest.raises(ValueError):
        _validated_num_batches(
            total_num_input_files=2, files_per_batch=6, num_batches=None
        )
    with pytest.raises(ValueError):
        _validated_num_batches(
            total_num_input_files=2, files_per_batch=1, num_batches=3
        )


def test__select_batch_timesteps():
    batched_times = _select_batch_timesteps(
        timesteps=[str(i) for i in range(12)],
        files_per_batch=3,
        num_batches=4,
        random_seed=0,
    )
    assert len(batched_times) == 4
    for batch in batched_times:
        assert len(batch) == 3
    with pytest.raises(Exception):
        _select_batch_timesteps(
            timesteps=[str(i) for i in range(12)],
            files_per_batch=3,
            num_batches=5,
            random_seed=0,
        )

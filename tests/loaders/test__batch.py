import os
import pytest
import synth
import xarray as xr

from fv3net.regression.loaders._batch import (
    _mapper_to_batches,
    _load_batch,
    _validated_num_batches,
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
    batched_data_sequence = _mapper_to_batches(
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
            _validated_num_batches(total_times, times_per_batch, num_batches,)
            == valid_num_batches
        )
    else:
        with pytest.raises(ValueError):
            _validated_num_batches(
                total_times, times_per_batch, num_batches,
            )

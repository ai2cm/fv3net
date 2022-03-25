import os
from loaders._utils import SAMPLE_DIM_NAME
import pytest
import synth
import xarray as xr
import numpy as np
import loaders
import loaders.mappers
import cftime
from loaders.batches._batch import (
    batches_from_mapper,
    _get_batch,
)

DATA_VARS = ["air_temperature", "specific_humidity"]
Z_DIM_SIZE = 79


class MockDatasetMapper:
    def __init__(self, schema: synth.DatasetSchema):
        self._schema = schema
        self._keys = [f"2000050{i+1}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = synth.generate(self._schema).drop("initial_time")
        return ds

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


@pytest.fixture(params=["MockDatasetMapper", "MultiDatasetMapper"])
def mapper(request, datadir):
    one_step_zarr_schema = "one_step_zarr_schema.json"
    # uses the one step schema but final mapper
    # functions the same for all data sources
    with open(os.path.join(datadir, one_step_zarr_schema)) as f:
        schema = synth.load(f)
    mapper = MockDatasetMapper(schema)
    if request.param == "MockDatasetMapper":
        return mapper
    elif request.param == "MultiDatasetMapper":
        return loaders.mappers.MultiDatasetMapper([mapper, mapper, mapper])
    else:
        raise ValueError("Invalid mapper type provided.")


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test__get_batch(mapper):
    ds = _get_batch(mapper=mapper, keys=mapper.keys(),)
    assert len(ds["time"]) == 4


def test_batches_from_mapper(mapper):
    batched_data_sequence = batches_from_mapper(
        mapper, DATA_VARS, timesteps_per_batch=2, needs_grid=False,
    )
    ds = batched_data_sequence[0]
    original_data_dims = {name: ds[name].dims for name in ds}
    original_dim_lengths = {dim: len(dim) for dim in ds.dims}
    assert len(batched_data_sequence) == 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)
        for name in batch.data_vars.keys():
            assert set(batch[name].dims) == set(original_data_dims[name])
        for dim in batch.dims:
            assert len(dim) == original_dim_lengths[dim]


@pytest.mark.parametrize(
    "total_times,times_per_batch,valid_num_batches", [(3, 1, 3), (3, 2, 2), (3, 4, 1)]
)
def test_batches_from_mapper_timestep_list(
    mapper, total_times, times_per_batch, valid_num_batches
):
    timestep_list = list(mapper.keys())[:total_times]
    batched_data_sequence = batches_from_mapper(
        mapper,
        DATA_VARS,
        timesteps_per_batch=times_per_batch,
        timesteps=timestep_list,
        needs_grid=False,
    )
    assert len(batched_data_sequence) == valid_num_batches
    timesteps_used = sum(batched_data_sequence._args, ())  # flattens list
    assert set(timesteps_used).issubset(timestep_list)


def test__batches_from_mapper_invalid_times(mapper):
    invalid_times = list(mapper.keys())[:2] + ["20000101.000000", "20000102.000000"]
    with pytest.raises(ValueError):
        batches_from_mapper(
            mapper,
            DATA_VARS,
            timesteps_per_batch=2,
            timesteps=invalid_times,
            needs_grid=False,
        )


def test_diagnostic_batches_from_mapper(mapper):
    batched_data_sequence = batches_from_mapper(
        mapper, DATA_VARS, timesteps_per_batch=2, needs_grid=False,
    )
    assert len(batched_data_sequence) == len(mapper) // 2 + len(mapper) % 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


@pytest.mark.parametrize(
    "tiles",
    [
        pytest.param([1, 2, 3, 4, 5, 6], id="one-indexed"),
        pytest.param([0, 1, 2, 3, 4, 5], id="zero-indexed"),
    ],
)
def test_batches_from_mapper_different_indexing_conventions(tiles):
    n = 48
    ds = xr.Dataset(
        {"a": (["time", "tile", "y", "x"], np.zeros((1, 6, n, n)))},
        coords={"time": [cftime.DatetimeJulian(2016, 8, 1)], "tile": tiles},
    )
    mapper = loaders.mappers.XarrayMapper(ds)
    seq = batches_from_mapper(mapper, ["a", "lon"], res=f"c{n}")
    assert len(seq) == 1
    assert ds.a[0].size == seq[0].a.size


def get_dataset(fill_value: float, n_vars: int, n_dims: int):
    data_vars = {}
    dims = [f"dim_{i}" for i in range(n_dims)]
    # need enough columns/data for random sample tests
    shape = tuple(range(6, 6 + n_dims))
    for i in range(n_vars):
        data_vars[f"var_{i}"] = xr.DataArray(
            np.full(shape=shape, fill_value=fill_value), dims=dims
        )
    return xr.Dataset(data_vars=data_vars)


def get_mapper(n_keys: int, n_vars: int, n_dims: int):
    mapper = {}
    for i in range(n_keys):
        mapper[str(i)] = get_dataset(fill_value=float(i), n_vars=n_vars, n_dims=n_dims)
    return mapper


@pytest.mark.parametrize(
    "n_keys", [pytest.param(1, id="one_key"), pytest.param(3, id="multiple_keys")]
)
@pytest.mark.parametrize(
    "stacked_dims, total_dims",
    [
        pytest.param(1, 1, id="stack_only_dim"),
        pytest.param(3, 3, id="stack_all_dims"),
        pytest.param(1, 3, id="stack_one_dim"),
        pytest.param(3, 6, id="stack_multiple_dims"),
    ],
)
def test_batches_from_mapper_stacking(n_keys: int, stacked_dims: int, total_dims: int):
    mapper = get_mapper(n_keys=n_keys, n_vars=2, n_dims=total_dims)
    variable_names = list(list(mapper.values())[0].data_vars.keys())
    unstacked_dims = [f"dim_{i}" for i in range(stacked_dims, total_dims)]
    result = batches_from_mapper(
        mapper, variable_names=variable_names, unstacked_dims=unstacked_dims
    )
    assert len(result) == n_keys  # default is 1 key per batch
    for batch in result:
        for data in batch.data_vars.values():
            assert len(set(data.dims).intersection(unstacked_dims)) == len(
                unstacked_dims
            )
            assert len(data.dims) == len(unstacked_dims) + 1
            assert data.dims[0] == SAMPLE_DIM_NAME


def test_batches_from_mapper_stacked_data_is_shuffled():
    # the assertions of this test have a small chance to fail for a given random seed
    np.random.seed(0)
    mapper = get_mapper(n_keys=10, n_vars=1, n_dims=3)
    unstacked_dims = ["dim_2"]
    result = batches_from_mapper(
        mapper,
        variable_names=["var_0"],
        unstacked_dims=unstacked_dims,
        timesteps_per_batch=10,
        shuffle_samples=True,
    )
    assert len(result) == 1
    batch = result[0]
    # if sufficiently large, should contain samples from every timestep
    sufficiently_large_subset = batch["var_0"].isel({SAMPLE_DIM_NAME: slice(0, 200)})
    for i in range(10):
        assert np.sum(sufficiently_large_subset.values == i) > 0
    # if sufficiently small, should *not* contain samples from every timestep
    sufficiently_small_subset = batch["var_0"].isel({SAMPLE_DIM_NAME: slice(0, 10)})
    for i in range(10):
        if np.sum(sufficiently_small_subset.values == i) == 0:
            break
    else:
        raise ValueError(
            "all timesteps were present in the first 10 samples, which is suspicious"
        )


@pytest.mark.parametrize(
    "n_keys", [pytest.param(1, id="one_key"), pytest.param(3, id="multiple_keys")]
)
def test_batches_from_mapper_unstacked(n_keys: int):
    n_dims = 3
    mapper = get_mapper(n_keys=n_keys, n_vars=2, n_dims=n_dims)
    variable_names = list(list(mapper.values())[0].data_vars.keys())
    result = batches_from_mapper(
        mapper, variable_names=variable_names, unstacked_dims=None
    )
    assert len(result) == n_keys  # default is 1 key per batch
    expected_dims = ["time"] + [f"dim_{i}" for i in range(n_dims)]
    for batch in result:
        for data in batch.data_vars.values():
            assert list(data.dims) == expected_dims


@pytest.mark.parametrize("subsample_ratio", [0.8, 0.5])  # 1.0 is covered by other tests
def test_batches_from_mapper_subsample(subsample_ratio: float):
    np.random.seed(0)  # results are slightly seed-dependent
    mapper = get_mapper(n_keys=10, n_vars=1, n_dims=3)
    unstacked_dims = ["dim_2"]
    result = batches_from_mapper(
        mapper,
        variable_names=["var_0"],
        unstacked_dims=unstacked_dims,
        timesteps_per_batch=10,
        subsample_ratio=subsample_ratio,
    )
    assert len(result) == 1
    data: np.ndarray = result[0]["var_0"].values
    non_subsampled_result = batches_from_mapper(
        mapper,
        variable_names=["var_0"],
        unstacked_dims=unstacked_dims,
        timesteps_per_batch=10,
        subsample_ratio=1.0,
    )
    n_total_samples = non_subsampled_result[0]["var_0"].shape[0]
    assert int(n_total_samples * subsample_ratio) == data.shape[0]
    for i in range(10):
        assert np.sum(data == i) > 0  # data should sample from all keys

    # unlikely to get exactly the same number from each, unless there is a bug where we
    # subsample each timestep independently
    sample_counts = []
    for i in range(10):
        sample_counts.append(np.sum(data == i))
    assert not all(count == sample_counts[0] for count in sample_counts)


def test_batches_from_netcdf(tmpdir):
    saved_batches = []
    for i in range(5):
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(
                    np.random.uniform(size=[2, 3, 4]), dims=["dim0", "dim1", "dim2"],
                )
            }
        )
        ds.to_netcdf(os.path.join(tmpdir, f"{i}.nc"))
        saved_batches.append(ds)
    loaded_batches = loaders.batches_from_netcdf(
        path=str(tmpdir), variable_names=list(ds.data_vars)
    )
    for ds_saved, ds_loaded in zip(saved_batches, loaded_batches):
        xr.testing.assert_equal(ds_saved, ds_loaded)


def test_batches_from_mapper_stacked_data_is_not_shuffled():
    mapper = get_mapper(n_keys=10, n_vars=1, n_dims=3)
    unstacked_dims = ["dim_2"]
    result = batches_from_mapper(
        mapper,
        variable_names=["var_0"],
        unstacked_dims=unstacked_dims,
        timesteps_per_batch=10,
        shuffle_timesteps=False,
        shuffle_samples=False,
    )
    assert len(result) == 1
    batch = result[0]
    multiindex = batch._fv3net_sample.values
    sample_times = [sample[0] for sample in multiindex]
    sample_dim1 = [sample[2] for sample in multiindex[:5]]
    assert sample_times == sorted(sample_times)
    assert sample_dim1 == sorted(sample_dim1)


def test_batches_from_mapper_data_transform(mapper):
    batched_data_sequence = batches_from_mapper(
        mapper,
        ["Q1", "Q2", "Qm"],
        timesteps_per_batch=2,
        needs_grid=False,
        data_transforms=[
            {"name": "Q1_from_dQ1_pQ1"},
            {"name": "Q2_from_dQ2_pQ2"},
            {"name": "Qm_from_Q1_Q2"},
        ],
    )
    ds = batched_data_sequence[0]
    assert "Q1" in ds
    assert "Q2" in ds
    assert "Qm" in ds

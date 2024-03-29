from fv3fit._shared import SliceConfig, PackerConfig
from typing import Iterable
from fv3fit._shared.packer import (
    pack,
    unpack,
    _unique_dim_name,
    count_features,
    clip,
)
from fv3fit.keras._models.packer import Unpack
import pytest
import numpy as np
import xarray as xr

SAMPLE_DIM = "sample"
FEATURE_DIM = "z"

DIM_LENGTHS = {
    SAMPLE_DIM: 32,
    FEATURE_DIM: 8,
}


@pytest.fixture(
    params=["one_1d_var", "one_2d_var", "two_2d_vars", "1d_and_2d", "five_vars"]
)
def dims_list(request) -> Iterable[str]:
    if request.param == "one_1d_var":
        return [[SAMPLE_DIM]]
    elif request.param == "one_2d_var":
        return [[SAMPLE_DIM, FEATURE_DIM]]
    elif request.param == "two_2d_vars":
        return [
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
        ]
    elif request.param == "1d_and_2d":
        return [[SAMPLE_DIM, FEATURE_DIM], [SAMPLE_DIM]]
    elif request.param == "five_vars":
        return [
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
        ]


def get_array(dims: Iterable[str], value: float) -> np.ndarray:
    return np.zeros([DIM_LENGTHS[dim] for dim in dims]) + value


@pytest.fixture
def names(dims_list: Iterable[str]):
    return [f"var_{i}" for i in range(len(dims_list))]


@pytest.fixture
def dataset(names: Iterable[str], dims_list: Iterable[str]) -> xr.Dataset:
    return get_dataset(names, dims_list)


def get_dataset(names: Iterable[str], dims_list: Iterable[str]) -> xr.Dataset:
    data_vars = {}
    for i, (name, dims) in enumerate(zip(names, dims_list)):
        data_vars[name] = xr.DataArray(get_array(dims, i), dims=dims)
    ds = xr.Dataset(data_vars)
    if FEATURE_DIM in ds.dims:
        ds = ds.assign_coords({FEATURE_DIM: np.arange(DIM_LENGTHS[FEATURE_DIM])})
    return ds


@pytest.fixture
def array(names: Iterable[str], dims_list: Iterable[str]) -> np.ndarray:
    array_list = []
    for i, dims in enumerate(dims_list):
        array = get_array(dims, i)
        if len(dims) == 1:
            array = array[:, None]
        array_list.append(array)
    return np.concatenate(array_list, axis=1)


def test_direct_unpack_layer():
    names = ["a", "b"]
    n_features = {"a": 4, "b": 4}
    packed = np.random.randn(3, 8)
    a = packed[:, :4]
    b = packed[:, 4:]
    unpack_layer = Unpack(pack_names=names, n_features=n_features, feature_dim=1)
    unpacked = unpack_layer(packed)
    np.testing.assert_array_almost_equal(unpacked[0], a)
    np.testing.assert_array_almost_equal(unpacked[1], b)


stacked_dataset = xr.Dataset({"a": xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["sample"])})


def test__unique_dim_name():
    with pytest.raises(ValueError):
        _unique_dim_name(stacked_dataset, "feature_sample")


def test_sklearn_pack(dataset: xr.Dataset, array: np.ndarray):
    packed_array, _ = pack(dataset, ["sample"])
    np.testing.assert_almost_equal(packed_array, array)


def test_sklearn_unpack(dataset: xr.Dataset):
    packed_array, feature_index = pack(dataset, ["sample"])
    unpacked_dataset = unpack(packed_array, ["sample"], feature_index=feature_index)
    xr.testing.assert_allclose(unpacked_dataset, dataset)


def test_sklearn_pack_unpack_with_clipping(dataset: xr.Dataset):
    name = list(dataset.data_vars)[0]
    if FEATURE_DIM in dataset[name].dims:
        pack_config = PackerConfig({name: SliceConfig(3, None)})
        packed_array, feature_index = pack(dataset, [SAMPLE_DIM], pack_config)
        unpacked_dataset = unpack(
            packed_array, [SAMPLE_DIM], feature_index=feature_index
        )
        expected = {}
        for k in dataset:
            da = dataset[k].copy(deep=True)
            slice_config = pack_config.clip.get(k, SliceConfig())
            da = da.isel({da.dims[-1]: slice_config.slice})
            expected[k] = da
        xr.testing.assert_allclose(unpacked_dataset, xr.Dataset(expected))


def test_clip(dataset: xr.Dataset):
    name = list(dataset.data_vars)[0]
    if FEATURE_DIM in dataset[name].dims:
        indices = {name: SliceConfig(4, 8)}
        clipped_data = clip(dataset, indices)
        da: xr.Dataset = dataset[name].assign_coords(
            {dim: range(dataset.sizes[dim]) for dim in dataset[name].dims}
        )
        slice_config = indices[name]
        expected_da = da.isel({da.dims[-1]: slice_config.slice})
        xr.testing.assert_identical(clipped_data[name], expected_da)


def test_clip_differing_slices():
    ds = get_dataset(
        ["var1", "var2"], [[SAMPLE_DIM, FEATURE_DIM], [SAMPLE_DIM, FEATURE_DIM]]
    )
    clip_config = {
        "var1": SliceConfig(4, 8),
        "var2": SliceConfig(None, 6, 2),
    }
    clipped_data = clip(ds, clip_config)
    for name in ds:
        da = ds[name].assign_coords(
            {dim: range(ds.sizes[dim]) for dim in ds[name].dims}
        )
        slice_config = clip_config[name]
        expected_da = da.isel({da.dims[-1]: slice_config.slice})
        xr.testing.assert_identical(clipped_data[name], expected_da)


def test_count_features():
    SAMPLE_DIM_NAME = "axy"
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.zeros([10]), dims=[SAMPLE_DIM_NAME]),
            "b": xr.DataArray(np.zeros([10, 1]), dims=[SAMPLE_DIM_NAME, "b_dim"]),
            "c": xr.DataArray(np.zeros([10, 5]), dims=[SAMPLE_DIM_NAME, "c_dim"]),
        }
    )
    names = list(ds.data_vars.keys())
    assert len(names) == 3
    index = ds.to_stacked_array(
        "feature", [SAMPLE_DIM_NAME], variable_dim="var"
    ).indexes["feature"]
    out = count_features(index, variable_dim="var")
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5

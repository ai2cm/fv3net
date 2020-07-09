from fv3net.regression.keras._packer import ArrayPacker
from typing import Iterable
import pytest
import numpy as np
import xarray as xr
import loaders

FEATURE_DIM = "z"

DIM_LENGTHS = {
    loaders.SAMPLE_DIM_NAME: 32,
    FEATURE_DIM: 8,
}


@pytest.fixture(params=["one_var", "two_2d_vars", "1d_and_2d"])
def dims_list(request) -> Iterable[str]:
    if request.param == "one_var":
        return [[loaders.SAMPLE_DIM_NAME, FEATURE_DIM]]
    elif request.param == "two_2d_vars":
        return [
            [loaders.SAMPLE_DIM_NAME, FEATURE_DIM],
            [loaders.SAMPLE_DIM_NAME, FEATURE_DIM],
        ]
    elif request.param == "1d_and_2d":
        return [[loaders.SAMPLE_DIM_NAME, FEATURE_DIM], [loaders.SAMPLE_DIM_NAME]]
    elif request.param == "five_vars":
        return [
            [loaders.SAMPLE_DIM_NAME, FEATURE_DIM],
            [loaders.SAMPLE_DIM_NAME],
            [loaders.SAMPLE_DIM_NAME, FEATURE_DIM],
            [loaders.SAMPLE_DIM_NAME],
            [loaders.SAMPLE_DIM_NAME, FEATURE_DIM],
        ]


def get_array(dims: Iterable[str], value: float) -> np.ndarray:
    return np.zeros([DIM_LENGTHS[dim] for dim in dims]) + value


@pytest.fixture
def names(dims_list: Iterable[str]):
    return [f"var_{i}" for i in range(len(dims_list))]


@pytest.fixture
def dataset(names: Iterable[str], dims_list: Iterable[str]) -> xr.Dataset:
    data_vars = {}
    for i, (name, dims) in enumerate(zip(names, dims_list)):
        data_vars[name] = xr.DataArray(get_array(dims, i), dims=dims)
    return xr.Dataset(
        data_vars, coords={FEATURE_DIM: np.arange(DIM_LENGTHS[FEATURE_DIM])}
    )


@pytest.fixture
def array(names: Iterable[str], dims_list: Iterable[str]) -> np.ndarray:
    array_list = []
    for i, dims in enumerate(dims_list):
        array = get_array(dims, i)
        if len(dims) == 1:
            array = array[:, None]
        array_list.append(array)
    return np.concatenate(array_list, axis=1)


@pytest.fixture
def packer(names: Iterable[str]) -> ArrayPacker:
    return ArrayPacker(names)


def test_to_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    result = packer.to_array(dataset)
    np.testing.assert_array_equal(result, array)


def test_to_dataset(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_dataset(array)
    xr.testing.assert_equal(result, dataset)


def test_unpack_before_pack_raises(packer: ArrayPacker, array: np.ndarray):
    with pytest.raises(RuntimeError):
        packer.to_dataset(array)


def test_repack_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_array(packer.to_dataset(array))
    np.testing.assert_array_equal(result, array)

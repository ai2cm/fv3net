from fv3fit._shared import ArrayPacker
from typing import Iterable
from fv3fit._shared.packer import unpack_matrix
import pytest
import numpy as np
import xarray as xr

SAMPLE_DIM = "sample"
FEATURE_DIM = "z"

DIM_LENGTHS = {
    SAMPLE_DIM: 32,
    FEATURE_DIM: 8,
}


@pytest.fixture(params=["one_var", "two_2d_vars", "1d_and_2d"])
def dims_list(request) -> Iterable[str]:
    if request.param == "one_var":
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
    return ArrayPacker(SAMPLE_DIM, names)


def test_to_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    result = packer.to_array(dataset)
    np.testing.assert_array_equal(result, array)


def test_to_dataset(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_dataset(array)
    # to_dataset does not preserve coordinates
    xr.testing.assert_equal(result, dataset)


def test_unpack_before_pack_raises(packer: ArrayPacker, array: np.ndarray):
    with pytest.raises(RuntimeError):
        packer.to_dataset(array)


def test_repack_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_array(packer.to_dataset(array))
    np.testing.assert_array_equal(result, array)


def test_unpack_matrix():
    nz = 10

    in_ = xr.Dataset({"a": (["x", "z"], np.ones((1, nz))), "b": (["x"], np.ones((1)))})
    out = xr.Dataset(
        {"c": (["x", "z"], np.ones((1, nz))), "d": (["x", "z"], np.ones((1, nz)))}
    )

    x_packer = ArrayPacker("x", pack_names=["a", "b"])
    y_packer = ArrayPacker("x", pack_names=["c", "d"])

    in_packed = x_packer.to_array(in_)
    out_packed = y_packer.to_array(out)

    matrix = np.outer(out_packed.squeeze(), in_packed.squeeze())
    jacobian = unpack_matrix(x_packer, y_packer, matrix)

    assert isinstance(jacobian[("a", "c")], xr.DataArray)
    assert jacobian[("a", "c")].dims == ("c", "a")
    assert isinstance(jacobian[("a", "d")], xr.DataArray)
    assert jacobian[("a", "d")].dims == ("d", "a")
    assert isinstance(jacobian[("b", "c")], xr.DataArray)
    assert jacobian[("b", "c")].dims == ("c", "b")
    assert isinstance(jacobian[("b", "d")], xr.DataArray)
    assert jacobian[("b", "d")].dims == ("d", "b")

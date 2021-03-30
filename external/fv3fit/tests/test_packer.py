from fv3fit._shared import ArrayPacker
from typing import Iterable
from fv3fit._shared.packer import (
    unpack_matrix,
    pack,
    unpack,
    Unpack,
    _unique_dim_name,
    count_features,
)
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


@pytest.fixture
def packer(names: Iterable[str]) -> ArrayPacker:
    return ArrayPacker(SAMPLE_DIM, names)


def test_to_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    result = packer.to_array(dataset)
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize(
    "dims_list", ["two_2d_vars", "1d_and_2d", "five_vars"], indirect=True
)
def test_pack_layer(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    inputs = [dataset[name].values for name in packer.pack_names]
    for i, value in enumerate(inputs):
        if len(value.shape) == 1:
            inputs[i] = value[:, None]
    result = packer.pack_layer()(inputs)
    np.testing.assert_array_equal(result, array)


def test_to_dataset(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_dataset(array)
    # to_dataset does not preserve coordinates
    xr.testing.assert_equal(result, dataset.drop(dataset.coords.keys()))


@pytest.mark.parametrize(
    "dims_list", ["two_2d_vars", "1d_and_2d", "five_vars"], indirect=True
)
def test_packer_unpack_layer(
    packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray
):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.unpack_layer(feature_dim=1)(array)
    # to_dataset does not preserve coordinates
    for name, array in zip(packer.pack_names, result):
        if array.shape[1] == 1:
            array = array[:, 0]
        np.testing.assert_array_equal(array, dataset[name])


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


stacked_dataset = xr.Dataset({"a": xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["sample"])})


def test__unique_dim_name():
    with pytest.raises(ValueError):
        _unique_dim_name(stacked_dataset, "feature_sample")


def test_sklearn_pack(dataset: xr.Dataset, array: np.ndarray):
    packed_array, _ = pack(dataset, "sample")
    np.testing.assert_almost_equal(packed_array, array)


def test_sklearn_unpack(dataset: xr.Dataset):
    packed_array, feature_index = pack(dataset, "sample")
    unpacked_dataset = unpack(packed_array, "sample", feature_index)
    xr.testing.assert_allclose(unpacked_dataset, dataset)


def test_count_features_2d():
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
    out = count_features(names, ds, sample_dim_name=SAMPLE_DIM_NAME)
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5


def get_3d_dataset(sample_dim_name):
    TIME_DIM_NAME = "fhds"
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.zeros([10, 2]), dims=[sample_dim_name, TIME_DIM_NAME]),
            "b": xr.DataArray(
                np.zeros([10, 2, 1]), dims=[sample_dim_name, TIME_DIM_NAME, "b_dim"]
            ),
            "c": xr.DataArray(
                np.zeros([10, 2, 5]), dims=[sample_dim_name, TIME_DIM_NAME, "c_dim"]
            ),
        }
    )
    return ds


def test_count_features_3d():
    SAMPLE_DIM_NAME = "axy"
    ds = get_3d_dataset(SAMPLE_DIM_NAME)
    names = list(ds.data_vars.keys())
    assert len(names) == 3
    out = count_features(names, ds, sample_dim_name=SAMPLE_DIM_NAME, is_3d=True)
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5


def test_repack_3d_array():
    # don't need to handle data with singleton feature dimensions, at the moment
    dataset = get_3d_dataset(SAMPLE_DIM).drop_vars("b")
    packer = ArrayPacker(SAMPLE_DIM, ["a", "c"])
    array = packer.to_array(
        dataset, is_3d=True
    )  # must pack first to know dimension lengths
    with pytest.raises(NotImplementedError):
        packer.to_dataset(array)

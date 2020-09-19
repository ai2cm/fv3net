from fv3fit._shared import ArrayPacker
from typing import Iterable, Optional, Mapping
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


def get_array(dims: Iterable[str], var_value: float) -> np.ndarray:
    if FEATURE_DIM in dims:
        array = np.zeros([DIM_LENGTHS[dim] for dim in dims]) + np.arange(
            var_value, var_value + DIM_LENGTHS[FEATURE_DIM]
        )
    else:
        array = np.zeros([DIM_LENGTHS[dim] for dim in dims]) + var_value
    return array


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
def array(dims_list: Iterable[str]) -> np.ndarray:
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
    xr.testing.assert_equal(result, dataset.drop(dataset.coords.keys()))


def test_unpack_before_pack_raises(packer: ArrayPacker, array: np.ndarray):
    with pytest.raises(RuntimeError):
        packer.to_dataset(array)


def test_repack_array(packer: ArrayPacker, dataset: xr.Dataset, array: np.ndarray):
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_array(packer.to_dataset(array))
    np.testing.assert_array_equal(result, array)


@pytest.fixture(
    params=[None, "feature_dim_0_4", "feature_dim_4_end", "feature1_only_4_end"]
)
def feature_dim_slices_args(request) -> slice:
    if request.param == "feature_dim_0_4":
        feature_dim_slice = {"var_0": (None, 4), "var_1": (None, 4)}
    elif request.param == "feature_dim_4_end":
        feature_dim_slice = {"var_0": (4, None), "var_1": (4, None)}
    elif request.param == "feature1_only_4_end":
        feature_dim_slice = {"var_0": (None,), "var_1": (4, None)}
    elif request.param is None:
        feature_dim_slice = None
    return feature_dim_slice


@pytest.fixture
def packer_with_feature_slices(
    names: Iterable[str],
    feature_dim_slices_args: Optional[Mapping[str, Iterable[Optional[int]]]],
) -> ArrayPacker:
    return ArrayPacker(SAMPLE_DIM, names, feature_dim_slices=feature_dim_slices_args)


@pytest.fixture
def array_with_feature_slices(
    dims_list: Iterable[str],
    names: Iterable[str],
    feature_dim_slices_args: Optional[Mapping[str, Iterable[Optional[int]]]],
) -> np.ndarray:
    array_list = []
    feature_dim_slices_args = feature_dim_slices_args or {
        name: (None,) for name in names
    }
    for i, (name, dims) in enumerate(zip(names, dims_list)):
        array = get_array(dims, i)
        if len(dims) == 1:
            array = array[:, None]
        elif len(dims) == 2:
            array = array[:, slice(*feature_dim_slices_args[name])]
        array_list.append(array)
    return np.concatenate(array_list, axis=1)


def test_to_array_with_feature_slice(
    packer_with_feature_slices: ArrayPacker,
    dataset: xr.Dataset,
    array_with_feature_slices: np.ndarray,
):
    result = packer_with_feature_slices.to_array(dataset)
    print(result)
    print(array_with_feature_slices)
    np.testing.assert_array_equal(result, array_with_feature_slices)

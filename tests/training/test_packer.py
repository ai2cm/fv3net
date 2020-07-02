from fv3net.regression.keras import ArrayPacker
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
def dims_list(request):
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


def get_array(dims, value):
    return np.zeros([DIM_LENGTHS[dim] for dim in dims]) + value


@pytest.fixture
def names(dims_list):
    return [f"var_{i}" for i in range(len(dims_list))]


@pytest.fixture
def dataset(names, dims_list):
    data_vars = {}
    for i, (name, dims) in enumerate(zip(names, dims_list)):
        data_vars[name] = xr.DataArray(get_array(dims, i), dims=dims)
    return xr.Dataset(
        data_vars, coords={FEATURE_DIM: np.arange(DIM_LENGTHS[FEATURE_DIM])}
    )


@pytest.fixture
def array(names, dims_list):
    array_list = []
    for i, dims in enumerate(dims_list):
        array = get_array(dims, i)
        if len(dims) == 1:
            array = array[:, None]
        array_list.append(array)
    return np.concatenate(array_list, axis=1)


@pytest.fixture
def packer(names):
    return ArrayPacker(names)


def test_pack(packer, dataset, array):
    result = packer.pack(dataset)
    np.testing.assert_array_equal(result, array)


def test_unpack(packer, dataset, array):
    print(dataset)
    packer.pack(dataset)  # must pack first to know dimension lengths
    result = packer.unpack(array)
    print(result)
    xr.testing.assert_equal(result, dataset)


def test_unpack_before_pack_raises(packer, array):
    with pytest.raises(RuntimeError):
        packer.unpack(array)


def test_repack(packer, dataset, array):
    packer.pack(dataset)  # must pack first to know dimension lengths
    result = packer.pack(packer.unpack(array))
    np.testing.assert_array_equal(result, array)

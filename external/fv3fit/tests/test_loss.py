from typing import Iterable
from fv3fit.keras._models.loss import _pack_weights, _weighted_loss
from fv3fit._shared import ArrayPacker
import numpy as np
import xarray as xr
import pytest


SAMPLE_DIM = "sample"
FEATURE_DIM = "z"


@pytest.fixture
def names(request):
    return request.param


@pytest.fixture
def weights(request):
    return request.param


@pytest.fixture
def features(request):
    return request.param


@pytest.fixture
def dataset(features) -> xr.Dataset:
    data_vars = {}
    for name, n_features in features.items():
        if n_features == 1:
            data_vars[name] = ([SAMPLE_DIM], np.zeros([1]))
        else:
            data_vars[name] = (
                [SAMPLE_DIM, f"{name}_{FEATURE_DIM}"],
                np.zeros([1, n_features]),
            )
    return xr.Dataset(data_vars)


@pytest.fixture
def packer(names: Iterable[str], dataset: xr.Dataset) -> ArrayPacker:
    packer = ArrayPacker(SAMPLE_DIM, names)
    packer.to_array(dataset)  # must let packer know about array shapes
    return packer


@pytest.mark.parametrize(
    "names,weights,features,std,reference",
    [
        pytest.param(
            ["a"],
            {"a": 1.0, "b": 2.0},
            {"a": 1, "b": 1},
            np.array([2]),
            np.array([[0.5]]),
            id="one_scalar",
        ),
        pytest.param(
            ["b"],
            {"a": 1.0, "b": 3.0},
            {"a": 1, "b": 3},
            np.array([3.0, 3.0, 3.0]),
            np.array([[1.0, 1.0, 1.0]]),
            id="one_vector",
        ),
        pytest.param(
            ["a", "b"],
            {"a": 1.0, "b": 2.0},
            {"a": 1, "b": 1},
            np.array([2, 2]),
            np.array([[0.5, 1]]),
            id="two_scalars",
        ),
        pytest.param(
            ["a", "b"],
            {"a": 1.0, "b": 2.0},
            {"a": 2, "b": 1},
            np.array([2, 2, 2]),
            np.array([[0.5, 0.5, 1]]),
            id="one_scalar_one_vector",
        ),
        pytest.param(
            ["b", "a"],
            {"a": 1.0, "b": 2.0},
            {"a": 2, "b": 1},
            np.array([2, 2, 2]),
            np.array([[1, 0.5, 0.5]]),
            id="one_scalar_one_vector_reverse_order",
        ),
    ],
    indirect=["names", "weights", "features"],
)
def test_pack_weights(packer, std, weights, reference):
    result = _pack_weights(packer, std, **weights)
    np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    "weights, loss, y_true, y_pred, reference",
    [
        pytest.param(
            np.array([2.0]),
            lambda x, y: abs(x - y),
            np.array([0.0]),
            np.array([1.0]),
            2.0,
            id="double_single_feature_loss",
        ),
        pytest.param(
            np.array([0.5, 1.0, 2.0]),
            lambda x, y: sum(abs(x - y)),
            np.array([0.0, 0, 0]),
            np.array([1.0, 10, 100]),
            210.5,
            id="varying_weight_loss",
        ),
    ],
)
def test_weighted_loss(weights, loss, y_true, y_pred, reference):
    loss = _weighted_loss(weights, loss)
    result = loss(y_true, y_pred)
    np.testing.assert_array_equal(result, reference)

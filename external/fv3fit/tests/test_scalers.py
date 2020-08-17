import numpy as np
import pytest
import tempfile
from typing import Mapping, Sequence
import xarray as xr

from fv3fit._shared.scaler import (
    StandardScaler, ManualScaler, _create_scaling_array, get_mass_scaler)
from fv3fit._shared.packer import ArrayPacker

SAMPLE_DIM = "sample"
FEATURE_DIM = "z"


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize_then_denormalize(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.denormalize(scaler.normalize(X))
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.normalize(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_normalize_then_denormalize(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.denormalize(scaler.normalize(X))
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_normalize_then_denormalize_on_reloaded_scaler(n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.normalize(X)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    result = scaler.denormalize(result)
    np.testing.assert_almost_equal(result, X)


def _dataset_from_mapping(mapping: Mapping[str, Sequence[float]]):
    data_vars = {}
    for var, values in mapping.items():
        array = np.array(values)
        if array.shape == (1,):
            dims = [SAMPLE_DIM]
        else:
            dims = [SAMPLE_DIM, FEATURE_DIM]
            array = np.reshape(array, (1, -1))
        data_vars[var] = (dims, array)
    return xr.Dataset(data_vars)


@pytest.mark.parametrize(
    "output_values, delp_weights, \
        variable_scale_factors, sqrt_weights, expected_weights",
    [
        pytest.param(
            {"y0": [[0.0, 1]], "y1": [[2.0, 3]]},
            np.array([1.0, 2.0]),
            {"y0": 100},
            False,
            [[100, 200, 1.0, 2.0]],
            id="all vertical features, with scale factor",
        ),
        pytest.param(
            {"y0": [[0.0, 1]], "y1": [2]},
            np.array([1.0, 2.0]),
            None,
            False,
            [[1.0, 2.0, 1.0]],
            id="one scalar feature",
        ),
        pytest.param(
            {"y0": [0.0], "y1": [2]},
            np.array([1.0, 2.0]),
            None,
            False,
            [[1.0, 1.0]],
            id="all scalar features",
        ),
        pytest.param(
            {"y0": [[0.0, 6.0]], "y1": [[4.0, 18.0]]},
            np.array([4.0, 9.0]),
            {"y0": 100},
            True,
            [[20, 30, 2.0, 3.0]],
            id="sqrt delp, but not scale factors",
        ),
    ],
)
def test__create_scaling_array(
    output_values, delp_weights, variable_scale_factors, sqrt_weights, expected_weights
):
    ds = _dataset_from_mapping(output_values)
    packer = ArrayPacker(
        sample_dim_name=SAMPLE_DIM, pack_names=sorted(list(ds.data_vars))
    )
    _ = packer.to_array(ds)
    weights = _create_scaling_array(
        packer, delp_weights, variable_scale_factors, sqrt_weights,
    )
    np.testing.assert_almost_equal(weights, expected_weights)


def test_weight_scaler_denormalize():
    y = np.array([[0.0, 1.0, 2.0, 3.0]])
    weights = np.array([[1.0, 100.0, 1.0, 2.0]])
    scaler = ManualScaler(weights)
    result = scaler.denormalize(y)
    np.testing.assert_almost_equal(result, [[0.0, 0.01, 2.0, 1.5]])


def test_weight_scaler_normalize():
    y = np.array([[0.0, 0.01, 2.0, 1.5]])
    weights = np.array([[1.0, 100.0, 1.0, 2.0]])
    scaler = ManualScaler(weights)
    result = scaler.normalize(y)
    np.testing.assert_almost_equal(result, [[0.0, 1.0, 2.0, 3.0]])


def test_weight_scaler_normalize_then_denormalize_on_reloaded_scaler():
    y = np.random.uniform(0, 10, 10)
    weights = np.random.uniform(0, 100, 10)
    scaler = ManualScaler(weights)
    result = scaler.normalize(y)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    result = scaler.denormalize(result)
    np.testing.assert_almost_equal(result, y)


def test_get_mass_scaler():
    ds = _dataset_from_mapping( {"y0": [[0., 3.]], "y1": [[2., 6.]]})
    packer = ArrayPacker(
        sample_dim_name=SAMPLE_DIM, pack_names=sorted(list(ds.data_vars))
    )
    y = packer.to_array(ds)
    delp = np.array([1., 4.])
    scale_factors = {"y0": 100}
    scaler = get_mass_scaler(
        packer,
        delp,
        scale_factors,
        True,
    )
    expected_normalized = [[0., 15., 2., 3.]]
    normalized = scaler.normalize(y)
    np.testing.assert_almost_equal(normalized, expected_normalized)

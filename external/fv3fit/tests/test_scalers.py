import numpy as np
import pytest
import tempfile
from typing import Mapping, Sequence
import xarray as xr

from fv3fit._shared.scaler import (
    dumps,
    loads,
    StandardScaler,
    ManualScaler,
    _create_scaling_array,
    get_mass_scaler,
)
from fv3fit._shared.packer import ArrayPacker

SAMPLE_DIM = "sample"
FEATURE_DIM = "z"
SEED = 1


def test_standard_scaler_not_fit_before_call():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.normalize(np.array([0.0, 1.0]))
    with pytest.raises(RuntimeError):
        scaler.denormalize(np.array([0.0, 1.0]))


@pytest.mark.parametrize("std_epsilon", [(1e-12), (1e-8)])
def test_standard_scaler_constant_scaling(std_epsilon):
    scaler = StandardScaler(std_epsilon)
    const = 10.0
    constant_feature = np.array([const for i in range(5)])
    varying_feature = np.array([i for i in range(5)])
    y = np.vstack([varying_feature, constant_feature, constant_feature * 2.0]).T
    scaler.fit(y)
    assert (scaler.std[1:] == std_epsilon).all()
    normed_sample = scaler.normalize(np.array([3.0, const, const * 2.0]))
    assert (normed_sample[1:] == 0.0).all()
    denormed_sample = scaler.denormalize(np.array([3.0, 0.0, 0.0]))
    assert denormed_sample[1] == const
    assert denormed_sample[2] == const * 2.0


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize_then_denormalize(n_samples, n_features):
    scaler = StandardScaler()
    np.random.seed(SEED)
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.denormalize(scaler.normalize(X))
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize(n_samples, n_features):
    np.random.seed(SEED)
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.normalize(X)
    np.testing.assert_almost_equal(np.mean(result, axis=0), 0)
    np.testing.assert_almost_equal(np.std(result, axis=0), 1)


@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_normalize_then_denormalize_on_reloaded_scaler(n_samples, n_features):
    np.random.seed(SEED)
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
            [[100 / 3, 200 / 3, 1.0 / 3, 2.0 / 3]],
            id="all vertical features, with scale factor",
        ),
        pytest.param(
            {"y0": [[0.0, 1]], "y1": [2]},
            np.array([1.0, 2.0]),
            None,
            False,
            [[1.0 / 3.0, 2.0 / 3.0, 1.0]],
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
            [
                [
                    20 / np.sqrt(13),
                    30 / np.sqrt(13),
                    2.0 / np.sqrt(13),
                    3.0 / np.sqrt(13),
                ]
            ],
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
    np.random.seed(SEED)
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
    ds = _dataset_from_mapping({"y0": [[0.0, 3.0]], "y1": [[2.0, 6.0]]})
    packer = ArrayPacker(
        sample_dim_name=SAMPLE_DIM, pack_names=sorted(list(ds.data_vars))
    )
    y = packer.to_array(ds)
    delp = np.array([1.0, 0.25])
    scale_factors = {"y0": 100}
    scaler = get_mass_scaler(packer, delp, scale_factors, True,)
    expected_normalized = [
        [0.0, 15 / np.sqrt(1.25), 2 / np.sqrt(1.25), 3 / np.sqrt(1.25)]
    ]
    normalized = scaler.normalize(y)
    np.testing.assert_almost_equal(normalized, expected_normalized)


standard_scaler = StandardScaler()
standard_scaler.fit(np.ones((10, 1)))


@pytest.mark.parametrize("scaler", [ManualScaler(np.array([0.5])), standard_scaler])
def test_dump_load_manual_scaler(scaler):
    decoded = loads(dumps(scaler))
    # ensure data is unchanged by testing behavior
    in_ = np.array([10.0])
    np.testing.assert_equal(decoded.normalize(in_), scaler.normalize(in_))
    np.testing.assert_equal(decoded.denormalize(in_), scaler.denormalize(in_))

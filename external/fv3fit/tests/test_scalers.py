import numpy as np
import pytest
import tempfile

from fv3fit._shared.scaler import (
    dumps,
    loads,
    StandardScaler,
    ManualScaler,
)

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


standard_scaler = StandardScaler()
standard_scaler.fit(np.ones((10, 1)))


@pytest.mark.parametrize("scaler", [ManualScaler(np.array([0.5])), standard_scaler])
def test_dump_load_manual_scaler(scaler):
    decoded = loads(dumps(scaler))
    # ensure data is unchanged by testing behavior
    in_ = np.array([10.0])
    np.testing.assert_equal(decoded.normalize(in_), scaler.normalize(in_))
    np.testing.assert_equal(decoded.denormalize(in_), scaler.denormalize(in_))

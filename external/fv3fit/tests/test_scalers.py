import numpy as np
import pytest

from fv3fit._shared import StandardScaler, MassScaler



@pytest.mark.parametrize("n_samples, n_features", [(10, 1), (10, 5)])
def test_standard_scaler_normalize_then_denormalize(
        scaler, fit_args, n_samples, n_features):
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
def test_normalize_then_denormalize(scaler, n_samples, n_features):
    scaler = StandardScaler()
    X = np.random.uniform(0, 10, size=[n_samples, n_features])
    scaler.fit(X)
    result = scaler.denormalize(scaler.normalize(X))
    np.testing.assert_almost_equal(result, X)


@pytest.mark.parametrize(
    "output_var_order, output_values, delp_weights, variable_scale_factors, expected",
    [
        (["ft0, ft1"], {"ft0": [0., 1], "ft1": [2., 3]}, [1., 2.], {"feature0": 100}, [0, 50., 2., 1.5]),
        (["ft0, ft1"], {{"ft0": [0., 1], "ft1": [2]}, [1., 2.], None)
    ]
)
def test_mass_scaler_normalize(
        output_var_order, output_values, delp_weights, variable_scale_factors):
    output_var_feature_count = {var: len(output_values[var]) for var in output_values}
    n_features = sum(list(output_var_feature_count.values()))
    X = np.random.uniform(0, 10, n_features])
    expected = X / delp_weights
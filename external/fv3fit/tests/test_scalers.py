import numpy as np
import pytest
import tempfile

from fv3fit._shared import StandardScaler, MassScaler


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


@pytest.mark.parametrize(
    "output_var_order, output_values, delp_weights, \
        variable_scale_factors, sqrt_weights, expected",
    [
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0, 1], "y1": [2.0, 3]},
            [1.0, 2.0],
            {"y0": 100},
            False,
            [0, 50.0, 2.0, 1.5],
            id="all vertical features, with scale factor",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0, 1], "y1": [2]},
            [1.0, 2.0],
            None,
            False,
            [0.0, 0.5, 2.0],
            id="one scalar feature",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0], "y1": [2]},
            [1.0, 2.0],
            None,
            False,
            [0.0, 2.0],
            id="all scalar features",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0, 6.0], "y1": [4.0, 18.0]},
            [4.0, 9.0],
            None,
            True,
            [0.0, 2.0, 2.0, 6.0],
            id="sqrt weights",
        ),
    ],
)
def test_mass_scaler_normalize(
    output_var_order,
    output_values,
    delp_weights,
    variable_scale_factors,
    sqrt_weights,
    expected,
):
    output_var_feature_count = {var: len(output_values[var]) for var in output_values}
    y = np.concatenate([output_values[var] for var in output_var_order]).ravel()
    scaler = MassScaler()
    scaler.fit(
        output_var_order,
        output_var_feature_count,
        delp_weights,
        variable_scale_factors,
        sqrt_weights,
    )
    result = scaler.normalize(y)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "output_var_order, output_values, delp_weights, \
        variable_scale_factors, sqrt_weights, expected",
    [
        pytest.param(
            ["y0", "y1"],
            {"y0": [0, 50], "y1": [2.0, 1.5]},
            [1.0, 2.0],
            {"y0": 100},
            False,
            [0.0, 1.0, 2.0, 3.0],
            id="all vertical features, with scale factor",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0, 0.5], "y1": [2]},
            [1.0, 2.0],
            None,
            False,
            [0.0, 1.0, 2.0],
            id="one scalar feature",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0], "y1": [2]},
            [1.0, 2.0],
            None,
            False,
            [0.0, 2.0],
            id="all scalar features",
        ),
        pytest.param(
            ["y0", "y1"],
            {"y0": [0.0, 2.0], "y1": [2.0, 6.0]},
            [4.0, 9.0],
            None,
            True,
            [0.0, 6.0, 4.0, 18.0],
            id="sqrt weights",
        ),
    ],
)
def test_mass_scaler_denormalize(
    output_var_order,
    output_values,
    delp_weights,
    variable_scale_factors,
    sqrt_weights,
    expected,
):
    output_var_feature_count = {var: len(output_values[var]) for var in output_values}
    y = np.concatenate([output_values[var] for var in output_var_order]).ravel()
    scaler = MassScaler()
    scaler.fit(
        output_var_order,
        output_var_feature_count,
        delp_weights,
        variable_scale_factors,
        sqrt_weights,
    )
    result = scaler.denormalize(y)
    np.testing.assert_almost_equal(result, expected)


def test_mass_scaler_normalize_then_denormalize_on_reloaded_scaler():
    output_var_order = ["y0", "y1", "y2"]
    output_var_feature_count = {"y0": 3, "y1": 3, "y2": 1}
    y = np.random.uniform(0, 10, 7)
    delp_weights = np.random.uniform(0, 10, size=3)
    variable_scale_factors = {var: np.random.uniform(0, 10) for var in output_var_order}
    scaler = MassScaler()
    scaler.fit(
        output_var_order,
        output_var_feature_count,
        delp_weights,
        variable_scale_factors,
    )
    result = scaler.normalize(y)
    with tempfile.NamedTemporaryFile() as f_write:
        scaler.dump(f_write)
        with open(f_write.name, "rb") as f_read:
            scaler = scaler.load(f_read)
    result = scaler.denormalize(result)
    np.testing.assert_almost_equal(result, y)

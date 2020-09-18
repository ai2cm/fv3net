import pytest
from offline_ml_diags.compute_diags import _get_model_loader
from offline_ml_diags._model_loaders import load_sklearn_model, load_keras_model


@pytest.mark.parametrize(
    ["config", "valid", "expected_loader", "expected_kwargs"],
    [
        pytest.param({'model_type': 'sklearn_random_forest'}, True, load_sklearn_model, {}, id='sklearn_rf'),
        pytest.param({'model_type': 'DenseModel'}, True, load_keras_model, {"keras_model_type": "DenseModel"}, id='keras_dense_model'),
        pytest.param({'model_type': "A model that doesn't exist"}, False, None, {}, id='invalid'),
    ]
)
def test__get_model(config, valid, expected_loader, expected_kwargs):
    if valid:
        loader, kwargs = _get_model_loader(config)
        assert loader == expected_loader
        assert kwargs == expected_kwargs 
    else:
        with pytest.raises(AttributeError):
            loader, kargs = _get_model_loader(config)
            
from runtime.config import _get_ml_model_class
from fv3fit.keras._models import DenseModel, DummyModel
from fv3fit.sklearn import SklearnWrapper
import pytest


@pytest.fixture
def ml_config(request):
    model_type, keras_model_type = request.param
    if model_type == "keras":
        return {
            "model_type": model_type,
            "model_loader_kwargs": {"keras_model_type": keras_model_type},
        }
    elif model_type == "scikit_learn":
        return {"model_type": model_type}


@pytest.mark.parametrize(
    ["ml_config", "expected"],
    [
        pytest.param(("keras", "DenseModel"), DenseModel, id="keras_dense_model"),
        pytest.param(("keras", "DummyModel"), DummyModel, id="keras_dummy_model"),
        pytest.param(("scikit_learn", None), SklearnWrapper, id="sklearn_wrapper"),
    ],
    indirect=["ml_config"],
)
def test__get_ml_model_class(ml_config, expected):
    model_class = _get_ml_model_class(ml_config)
    assert model_class == expected

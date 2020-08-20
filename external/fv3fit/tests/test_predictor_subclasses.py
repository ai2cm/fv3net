from fv3fit._shared import Predictor
from fv3fit.keras import get_model_class
from fv3fit.sklearn import SklearnWrapper
import pytest

# additional public models from fv3fit should be registered here as they are created
# to ensure that they subclass the required Predictor abstract base class


@pytest.mark.parametrize("model_class_name", [("DenseModel"), ("DummyModel")])
def test_keras_public_Predictors(model_class_name):
    assert issubclass(get_model_class(model_class_name), Predictor)


@pytest.mark.parametrize("model_class", [(SklearnWrapper)])
def test_sklearn_public_Predictors(model_class):
    assert issubclass(model_class, Predictor)

import pytest

from fv3fit._shared.config import register_keras_estimator, get_keras_model


def test_get_model_raises_before_registering():
    with pytest.raises(KeyError):
        get_keras_model("MyModel")


def test_get_model_returns_after_registering():
    @register_keras_estimator("name")
    class MyModel:
        pass

    cls = get_keras_model("name")
    assert cls == MyModel


def test_register_raises_if_no_name_given():
    with pytest.raises(TypeError, match="remember to pass one"):

        @register_keras_estimator
        class MyModel:
            pass

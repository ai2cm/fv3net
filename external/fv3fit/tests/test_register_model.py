import pytest
import contextlib

from fv3fit._shared.config import (
    get_training_function,
    register_training_function,
    get_hyperparameter_class,
    TrainingConfig,
    ESTIMATORS,
)


class MyConfig(TrainingConfig):
    pass


@contextlib.contextmanager
def registration_context():
    original_estimators = {**ESTIMATORS}
    ESTIMATORS.clear()
    try:
        yield
    finally:
        ESTIMATORS.clear()
        ESTIMATORS.update(original_estimators)


def test_get_training_function_raises_before_registering():
    with registration_context():
        with pytest.raises(ValueError):
            get_training_function("MyModel")


def test_get_training_function_returns_after_registering():
    with registration_context():

        @register_training_function("name", MyConfig)
        def my_training_function(*args, **kwargs):
            pass

        func = get_training_function("name")
        assert func is my_training_function


def test_get_config_class_raises_before_registering():
    with registration_context():
        with pytest.raises(ValueError):
            get_training_function("MyModel")


def test_get_config_class_returns_after_registering():
    with registration_context():

        @register_training_function("name", MyConfig)
        def my_training_function(*args, **kwargs):
            pass

        cls = get_hyperparameter_class("name")
        assert cls is MyConfig


def test_register_raises_if_no_name_given():
    with registration_context():
        with pytest.raises(TypeError):

            @register_training_function
            def my_training_function(*args, **kwargs):
                pass

import pytest
import contextlib

from fv3fit._shared.config import (
    register_estimator,
    get_config_class,
    get_estimator_class,
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


def test_get_estimator_class_raises_before_registering():
    with registration_context():
        with pytest.raises(ValueError):
            get_estimator_class("MyModel")


def test_get_estimator_class_returns_after_registering():
    with registration_context():

        @register_estimator("name", MyConfig)
        class MyModel:
            pass

        cls = get_estimator_class("name")
        assert cls == MyModel


def test_get_config_class_raises_before_registering():
    with registration_context():
        with pytest.raises(ValueError):
            get_config_class("MyModel")


def test_get_config_class_returns_after_registering():
    with registration_context():

        @register_estimator("name", MyConfig)
        class MyModel:
            pass

        cls = get_config_class("name")
        assert cls == MyConfig


def test_register_raises_if_no_name_given():
    with registration_context():
        with pytest.raises(TypeError):

            @register_estimator
            class MyModel:
                pass

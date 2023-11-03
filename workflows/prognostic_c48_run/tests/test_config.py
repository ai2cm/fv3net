import dacite
import pytest
from runtime.config import get_model_urls, get_wrapper, UserConfig
from runtime.names import FV3GFS_WRAPPER
import dataclasses
from testing_utils import requires_fv3gfs_wrapper
from types import ModuleType

dummy_prescriber = {"dataset_key": "data_url", "variables": {"a": "a"}}


@pytest.mark.parametrize(
    "config, model_urls",
    [
        ({}, []),
        ({"scikit_learn": None, "prephysics": None}, []),
        ({"scikit_learn": None, "prephysics": [dummy_prescriber]}, []),
        (
            {
                "scikit_learn": {"model": ["ml_model_url"]},
                "prephysics": [dummy_prescriber],
            },
            ["ml_model_url"],
        ),
        (
            {
                "scikit_learn": {"model": ["ml_model_url"]},
                "prephysics": [
                    dummy_prescriber,
                    {"model": ["prephysics_model_0", "prephysics_model_1"]},
                ],
            },
            ["ml_model_url", "prephysics_model_0", "prephysics_model_1"],
        ),
    ],
)
def test_get_model_urls(config, model_urls):
    # Since this function is coupled to the UserConfig, check that test is in sync
    # with this class
    validated_config = dataclasses.asdict(
        dacite.from_dict(UserConfig, config, dacite.Config(strict=True))
    )
    assert set(get_model_urls(validated_config)) == set(model_urls)


@requires_fv3gfs_wrapper
def test_get_wrapper():
    config = UserConfig(wrapper=FV3GFS_WRAPPER)
    wrapper = get_wrapper(config)
    assert isinstance(wrapper, ModuleType)

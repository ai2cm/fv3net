import dacite
import pytest
from runtime.config import get_model_urls, get_wrapper, UserConfig
import dataclasses

from testing_utils import (
    has_fv3gfs_wrapper,
    has_shield_wrapper,
    requires_fv3gfs_wrapper,
    requires_shield_wrapper,
)


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


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("fv3gfs", marks=requires_fv3gfs_wrapper),
        pytest.param("shield", marks=requires_shield_wrapper),
    ],
)
def test_get_wrapper(model):
    config = UserConfig(model=model)
    get_wrapper(config)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "fv3gfs",
            marks=pytest.mark.skipif(
                has_fv3gfs_wrapper, reason="requires fv3gfs.wrapper NOT be installed"
            ),
        ),
        pytest.param(
            "shield",
            marks=pytest.mark.skipif(
                has_shield_wrapper, reason="requires shield.wrapper NOT be installed"
            ),
        ),
    ],
)
def test_get_wrapper_error(model):
    config = UserConfig(model=model)
    with pytest.raises(ImportError, match="required for running"):
        get_wrapper(config)

"""Integration tests that are useful for development, but perhaps too flaky for
CI.
"""
import pytest
import emulation.config


@pytest.mark.xfail
def test_combined_classifier():
    config = emulation.config.ModelConfig(
        path="gs://vcm-ml-experiments/microphysics-emulation/2022-06-28/gscond-routed-reg-v3/model.tf",  # noqa
        classifier_path="gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1/model.tf",  # noqa
    )
    assert config.build()

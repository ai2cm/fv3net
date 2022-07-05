"""Integration tests that are useful for development, but perhaps too flaky for
CI.
"""
import emulation.config


def test_combined_classifier():
    config = emulation.config.ModelConfig(
        path="gs://vcm-ml-experiments/microphysics-emulation/2022-07-02/gscond-routed-reg-v4/model.tf",  # noqa
        classifier_path="gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1/model.tf",  # noqa
        enforce_conservative=True,
    )
    assert config.build()

"""Integration tests that are useful for development, but perhaps too flaky for
CI.
"""
import emulation.config
from fv3fit.emulation.data.io import download_cached
import xarray
import numpy as np


def test_combined_classifier():
    model = "gs://vcm-ml-experiments/microphysics-emulation/2022-06-28/gscond-routed-reg-v3/model.tf"  # noqa
    config = emulation.config.ModelConfig(
        path=model,
        classifier_path="gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1/model.tf",  # noqa
        enforce_conservative=True,
    )
    nc_data = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4/test/state_20160203.014500_0.nc"  # noqa
    path = download_cached(nc_data)
    ds = xarray.open_dataset(path)
    state = {key: np.asarray(v).astype(np.float64).T for key, v in ds.items()}
    model = config.build()
    model.microphysics(state)

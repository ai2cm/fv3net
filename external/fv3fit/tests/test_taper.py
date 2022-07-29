import fsspec
import fv3fit
from fv3fit._shared.models import OutOfSampleModel
from fv3fit._shared.taper_function import (
    get_taper_function,
    taper_decay,
    taper_mask,
    taper_ramp,
)
from fv3fit.testing import ConstantOutputNoveltyDetector, ConstantOutputPredictor
import numpy as np
import os
import pytest
import tempfile
import xarray as xr
import yaml


@pytest.mark.slow
def test_masked_tapering():
    cutoff = 3
    taper = get_taper_function(taper_mask.__name__, {"cutoff": cutoff})
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper(novelty_score)
    np.testing.assert_almost_equal(taper_values, np.asarray([[1, 1, 0], [0, 0, 1]]))


@pytest.mark.slow
def test_ramp_tapering():
    ramp_min = 2
    ramp_max = 5
    taper = get_taper_function(
        taper_ramp.__name__, {"ramp_min": ramp_min, "ramp_max": ramp_max}
    )
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper(novelty_score)
    np.testing.assert_almost_equal(
        taper_values, np.asarray([[1, 2 / 3, 0], [0, 1 / 3, 1]])
    )


@pytest.mark.slow
def test_decay_tapering():
    threshold = 2
    rate = 0.5
    taper = get_taper_function(
        taper_decay.__name__, {"threshold": threshold, "rate": rate}
    )
    novelty_score = xr.DataArray([[1, 3, 5], [6, 4, 2]])
    taper_values = taper(novelty_score)
    np.testing.assert_almost_equal(
        taper_values, np.asarray([[1, 2 ** -1, 2 ** -3], [2 ** -4, 2 ** -2, 1]])
    )


def load_oos_model_with_tapering_config(tapering_config):
    base_model = ConstantOutputPredictor([], [])
    novelty_detector = ConstantOutputNoveltyDetector([])

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base")
        novelty_path = os.path.join(tmpdir, "novelty")
        fv3fit.dump(base_model, base_path)
        fv3fit.dump(novelty_detector, novelty_path)

        config = {
            "base_model_path": base_path,
            "novelty_detector_path": novelty_path,
        }
        if tapering_config is not None:
            config["tapering_function"] = tapering_config

        with fsspec.open(
            os.path.join(tmpdir, OutOfSampleModel._CONFIG_FILENAME), "w"
        ) as f:
            yaml.safe_dump(config, f)
        loaded_model = OutOfSampleModel.load(tmpdir)
        return loaded_model


@pytest.mark.slow
def test_taper_loading_default_config():
    loaded_model = load_oos_model_with_tapering_config(None)
    # tapering function is a mask with cutoff zero
    np.testing.assert_allclose(
        loaded_model.taper(xr.DataArray([-1e-5, 1e-5])), xr.DataArray([1, 0])
    )


@pytest.mark.slow
def test_taper_loader_ramp_config():
    tapering_config = {
        "name": taper_ramp.__name__,
        "ramp_min": -1,
        "ramp_max": 2,
    }
    loaded_model = load_oos_model_with_tapering_config(tapering_config)
    # tapering function is a ramp from -1 to 2
    np.testing.assert_allclose(
        loaded_model.taper(xr.DataArray([-1, 0.5, 2])), xr.DataArray([1, 0.5, 0])
    )

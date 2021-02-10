import pytest
import logging
from dataclasses import dataclass
import tempfile
import os
from typing import Optional

import fv3fit
from offline_ml_diags.compute_diags import main

logger = logging.getLogger(__name__)


@pytest.fixture
def model(training_batches) -> fv3fit.Estimator:
    model = fv3fit.keras.get_model(
        "DenseModel",
        "sample",
        ["air_temperature", "specific_humidity"],
        ["dQ1", "dQ2"],
        width=3,
        depth=2,
    )
    model.fit(training_batches)
    return model


@dataclass
class Args:
    model_path: str
    output_path: str
    grid: str
    timesteps_n_samples: Optional[int] = 2
    data_path: Optional[str] = None
    config_yml: Optional[str] = None
    timesteps_file: Optional[str] = None
    training: Optional[bool] = False
    snapshot_time: Optional[str] = None


@pytest.mark.parametrize("data_source_name", ["nudging_tendencies"], indirect=True)
def test_offline_diags_integration(
    model, train_config, data_source_path, grid_dataset_path,
):
    """
    Test the bash endpoint for computing offline diagnostics
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        model.dump(model_dir)
        train_config.data_path = data_source_path
        train_config.dump(model_dir)
        args = Args(model_dir, os.path.join(tmpdir, "offline_diags"), grid_dataset_path)
        main(args)

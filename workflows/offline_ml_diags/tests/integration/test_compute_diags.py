import pytest
import logging
import subprocess
import tempfile
import os

import fv3fit

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


@pytest.mark.parametrize("data_source_name", ["nudging_tendencies"], indirect=True)
def test_offline_diags_integration(
    model,
    train_config,
    # data_source_name: str,
    data_source_path,
    grid_dataset_path,
):
    """
    Test the bash endpoint for computing offline diagnostics
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        model.dump(model_dir)
        train_config.data_path = data_source_path
        train_config.dump(model_dir)
        subprocess.check_call(
            [
                "python",
                "-m",
                "offline_ml_diags.compute_diags",
                model_dir,
                os.path.join(tmpdir, "offline_diags"),
                "--timesteps-n-samples",
                "2",
                "--grid",
                grid_dataset_path,
            ]
        )

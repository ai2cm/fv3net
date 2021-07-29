import logging
from dataclasses import dataclass
import tempfile
import os
from typing import Optional
import synth
from synth import (  # noqa: F401
    grid_dataset,
    grid_dataset_path,
    dataset_fixtures_dir,
)

# TODO: refactor this code to use the public TrainingConfig and DataConfig
# classes from fv3fit instead of _ModelTrainingConfig
from fv3fit._shared.config import _ModelTrainingConfig as ModelTrainingConfig
import fv3fit
from offline_ml_diags.compute_diags import main
import pathlib
import pytest
import numpy as np

logger = logging.getLogger(__name__)


@pytest.fixture
def data_path(tmpdir):
    schema_path = pathlib.Path(__file__).parent / "data.zarr.json"

    with open(schema_path) as f:
        schema = synth.load(f)

    ranges = {"pressure_thickness_of_atmospheric_layer": synth.Range(0.99, 1.01)}
    ds = synth.generate(schema, ranges)

    ds.to_zarr(str(tmpdir), consolidated=True)
    return str(tmpdir)


batch_kwargs = {
    "needs_grid": False,
    "res": "c8_random_values",
    "timesteps_per_batch": 1,
    "mapping_function": "open_zarr",
    "timesteps": ["20160801.001500"],
    "mapping_kwargs": {},
}


@dataclass
class Args:
    model_path: str
    output_path: str
    grid: str
    timesteps_n_samples: Optional[int] = 2
    data_path: Optional[str] = None
    config_yml: Optional[str] = None
    timesteps_file: Optional[str] = None
    snapshot_time: Optional[str] = None


# TODO: refactor this test to directly call fv3fit.train as another main routine,
# instead of duplicating train logic above in `model` routine
def test_offline_diags_integration(data_path, grid_dataset_path):  # noqa: F811
    """
    Test the bash endpoint for computing offline diagnostics
    """
    train_config = ModelTrainingConfig(
        model_type="DenseModel",
        hyperparameters={"width": 3, "depth": 2},
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
        batch_function="batches_from_geodata",
        batch_kwargs=batch_kwargs,
        scaler_type="standard",
        scaler_kwargs={},
        additional_variables=[],
        random_seed=0,
        validation_timesteps=None,
        data_path=None,
    )
    trained_model = fv3fit.testing.ConstantOutputPredictor(
        "sample",
        input_variables=train_config.input_variables,
        output_variables=train_config.output_variables,
    )
    trained_model.set_outputs(dQ1=np.zeros([19]), dQ2=np.zeros([19]))
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        fv3fit.dump(trained_model, model_dir)
        train_config.data_path = data_path
        train_config.dump(model_dir)
        args = Args(model_dir, os.path.join(tmpdir, "offline_diags"), grid_dataset_path)
        main(args)

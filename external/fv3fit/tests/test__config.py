import os
import tempfile
from fv3fit._shared.config import ModelTrainingConfig

import pytest


config = ModelTrainingConfig(
    model_type="great_model",
    hyperparameters={"max_depth": 10},
    input_variables=["in0", "in1"],
    output_variables=["out0, out1"],
    batch_function="batches_from_mapper",
    batch_kwargs={"timesteps_per_batch": 1},
)


def test_dump_and_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config.dump(tmpdir)
        loaded = ModelTrainingConfig.load(os.path.join(tmpdir, "training_config.yml"))
        assert config.asdict() == loaded.asdict()


@pytest.mark.parametrize(
    "scaler_type, expected",
    [("mass", ["pressure_thickness_of_atmospheric_layer"]), (None, [])],
)
def test_config_variables(scaler_type, expected):
    config = ModelTrainingConfig(
        model_type="great_model",
        hyperparameters={"max_depth": 10},
        input_variables=["in0", "in1"],
        output_variables=["out0, out1"],
        batch_function="batches_from_mapper",
        batch_kwargs={"timesteps_per_batch": 1},
        scaler_type=scaler_type,
    )
    assert expected == config.additional_variables

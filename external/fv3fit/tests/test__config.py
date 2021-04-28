import os
import tempfile
from fv3fit._shared.config import ModelTrainingConfig


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

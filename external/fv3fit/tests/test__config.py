import inspect
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


def _attributes_to_dict(obj):
    attributes = inspect.getmembers(obj, lambda a: not (inspect.isroutine(a)))
    return {
        key: value
        for key, value in attributes
        if not (key.startswith("__") and key.endswith("__"))
    }


def test_dump_and_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config.dump(tmpdir)
        loaded = ModelTrainingConfig.load(os.path.join(tmpdir, "training_config.yml"))
        assert _attributes_to_dict(config) == _attributes_to_dict(loaded)

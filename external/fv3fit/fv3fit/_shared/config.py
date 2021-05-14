import dataclasses
import fsspec
import yaml
import os
from typing import Dict, Optional, Union, Sequence, List


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


KERAS_MODELS: Dict[str, type] = {}
SKLEARN_MODEL_TYPES = ["sklearn", "rf", "random_forest", "sklearn_random_forest"]


def get_keras_model(name):
    return KERAS_MODELS[name]


def register_keras_trainer(name: str):
    """
    Returns a decorator that will register the given class as a keras training
    class, which can be used in training configuration.
    """
    if not isinstance(name, str):
        raise TypeError(
            "keras trainer name must be string, remember to "
            "pass one when decorating @register_keras_trainer(name)"
        )

    def decorator(cls):
        KERAS_MODELS[name] = cls
        return cls

    return decorator


@dataclasses.dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info

    Attrs:
        model_type: sklearn model type or keras model class to initialize
        hyperparameters: arguments to pass to model class at initialization
            time
        input_variables: variables used as features
        output_variables: variables to predict
        batch_function: name of function from `fv3fit.batches` to use for
            loading batched data
        batch_kwargs: keyword arguments to pass to batch function
        data_path: location of training data to be loaded by batch function
        scaler_type: scaler to use for training
        scaler_kwargs: keyword arguments to pass to scaler initialization
        additional_variables: list of needed variables which are not inputs
            or outputs (e.g. pressure thickness if needed for scaling)
        random_seed: value to use to initialize randomness
        validation_timesteps: timestamps to use as validation samples
        save_model_checkpoints: whether to save a copy of the model at
            each epoch
        model_path: output location for final model
        timesteps_source: one of "timesteps_file",
            "sampled_outside_input_config", "input_config", "all_mapper_times"
    """

    model_type: str
    hyperparameters: dict
    input_variables: List[str]
    output_variables: List[str]
    batch_function: str
    batch_kwargs: dict
    data_path: Optional[str] = None
    scaler_type: str = "standard"
    scaler_kwargs: dict = dataclasses.field(default_factory=dict)
    additional_variables: List[str] = dataclasses.field(default_factory=list)
    random_seed: Union[float, int] = 0
    validation_timesteps: Sequence[str] = dataclasses.field(default_factory=list)
    save_model_checkpoints: bool = False
    model_path: str = ""
    timesteps_source: str = "timesteps_file"

    def __post_init__(self):
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)

    def asdict(self):
        return dataclasses.asdict(self)

    def dump(self, path: str, filename: str = None) -> None:
        dict_ = self.asdict()
        if filename is None:
            filename = MODEL_CONFIG_FILENAME
        with fsspec.open(os.path.join(path, filename), "w") as f:
            yaml.safe_dump(dict_, f)

    @classmethod
    def load(cls, path: str) -> "ModelTrainingConfig":
        with fsspec.open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return ModelTrainingConfig(**config_dict)


def load_training_config(model_path: str) -> ModelTrainingConfig:
    """Load training configuration information from a model directory URL.

    Note:
        This loads a file that you would get from using ModelTrainingConfig.dump
        with no filename argument, as is done by fv3fit.train. To ensure
        backwards compatibility, you should use this routine to load such
        a file instead of manually specifying the filename.
        The default filename may change in the future.
    Args:
        model_path: model dir dumped by fv3fit.dump
    Returns:
        dict: training config dict
    """
    config_path = os.path.join(model_path, MODEL_CONFIG_FILENAME)
    return ModelTrainingConfig.load(config_path)

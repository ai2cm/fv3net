import inspect
import yaml
import os
from typing import Iterable, Optional, Union, Sequence, List
import vcm


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    def __init__(
        self,
        data_path: str,
        model_type: str,
        hyperparameters: dict,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        batch_function: str,
        batch_kwargs: dict,
        scaler_type: str = "standard",
        scaler_kwargs: Optional[dict] = None,
        additional_variables: Optional[List[str]] = None,
        random_seed: Union[float, int] = 0,
        validation_timesteps: Optional[Sequence[str]] = None,
        save_model_checkpoints: bool = False,
    ):
        self.data_path = data_path
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.batch_function = batch_function
        self.batch_kwargs = batch_kwargs
        self.scaler_type = scaler_type
        self.scaler_kwargs: dict = scaler_kwargs or {}
        self.additional_variables: List[str] = additional_variables or []
        self.random_seed = random_seed
        self.validation_timesteps: Sequence[str] = validation_timesteps or []
        self.save_model_checkpoints = save_model_checkpoints
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)

    def dump(self, f):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        dict_ = {
            key: value
            for key, value in attributes
            if not (key.startswith("__") and key.endswith("__"))
        }
        yaml.safe_dump(dict_, f)


def load_model_training_config(config_path: str, data_path: str) -> ModelTrainingConfig:
    """

    Args:
        config_path: location of .yaml that contains config for model training

    Returns:
        ModelTrainingConfig object
    """
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    return ModelTrainingConfig(data_path, **config_dict)


def save_config_output(
    output_url: str, config: ModelTrainingConfig,
):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)

    with fs.open(config_url, "w") as f:
        config.dump(f)

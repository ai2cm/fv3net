import inspect
import yaml
import os
from typing import Iterable, Optional, Union, Sequence, List

from ._filesystem import put_dir, get_dir

DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        batch_function: str,
        batch_kwargs: dict,
        data_path: Optional[str] = None,
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

    def dump(self, path: str) -> None:
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        dict_ = {
            key: value
            for key, value in attributes
            if not (key.startswith("__") and key.endswith("__"))
        }
        dir_ = os.path.join(path, MODEL_CONFIG_FILENAME)
        with put_dir(dir_) as output_path:
            with open(output_path, "w") as f:
                yaml.safe_dump(dict_, f)
    
    @classmethod
    def load(cls, path: str) -> "ModelTrainingConfig":
        with get_dir(path) as config_path:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        return ModelTrainingConfig(**config_dict)

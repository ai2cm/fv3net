import yaml
import dataclasses
from typing import Iterable, Optional


@dataclasses.dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    model_type: str
    hyperparameters: dict
    input_variables: Iterable[str]
    output_variables: Iterable[str]
    batch_function: str
    batch_kwargs: dict
    scaler_type: Optional[str] = None
    scaler_kwargs: Optional[dict] = None
    additional_variables: Optional[Iterable[str]] = None


def load_model_training_config(config_path: str) -> ModelTrainingConfig:
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
    return ModelTrainingConfig(**config_dict)

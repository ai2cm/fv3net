import yaml
import dataclasses
import os
from typing import Iterable, Optional, Mapping
import vcm


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


@dataclasses.dataclass
class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    data_path: str
    model_type: str
    hyperparameters: dict
    input_variables: Iterable[str]
    output_variables: Iterable[str]
    batch_function: str
    batch_kwargs: dict
    scaler_type: str = "standard"
    scaler_kwargs: Mapping = dataclasses.field(default_factory=dict)
    additional_variables: Optional[Iterable[str]] = None

    def __post_init__(self):
        self.additional_variables = self.additional_variables or []
        self.scaler_kwargs = self.scaler_kwargs or {}
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)

    def dump(self, f):
        dict_ = dataclasses.asdict(self)
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
    return ModelTrainingConfig({**config_dict, "data_path": data_path})


def save_config_output(
    output_url: str, config: ModelTrainingConfig,
):
    fs = vcm.cloud.fsspec.get_fs(output_url)
    fs.makedirs(output_url, exist_ok=True)
    config_url = os.path.join(output_url, MODEL_CONFIG_FILENAME)

    with fs.open(config_url, "w") as f:
        config.dump(f)

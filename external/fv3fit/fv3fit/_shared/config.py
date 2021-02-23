import fsspec
import inspect
import yaml
import os
from typing import Optional, Union, Sequence, List


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


class ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info
    """

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict,
        input_variables: List[str],
        output_variables: List[str],
        batch_function: str,
        batch_kwargs: dict,
        data_path: Optional[str] = None,
        scaler_type: str = "standard",
        scaler_kwargs: Optional[dict] = None,
        additional_variables: Optional[List[str]] = None,
        random_seed: Union[float, int] = 0,
        validation_timesteps: Optional[Sequence[str]] = None,
        save_model_checkpoints: Optional[bool] = False,
        model_path: Optional[str] = None,
        timesteps_source: Optional[str] = None,
    ):
        """
            Initialize the configuration class.

            Args:
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
        self.timesteps_source = timesteps_source
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)
        self.model_path = model_path

    def dump(self, path: str, filename: str = None) -> None:
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        dict_ = {
            key: value
            for key, value in attributes
            if not (key.startswith("__") and key.endswith("__"))
        }
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

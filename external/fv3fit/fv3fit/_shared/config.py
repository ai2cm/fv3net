import dataclasses
import fsspec
import yaml
import os
from typing import Optional, Tuple, Union, Sequence, List, Type, Dict
import xarray as xr
from .predictor import Estimator

from loaders import batches


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


# TODO: delete this routine by refactoring the tests to no longer depend on it
def get_keras_model(name):
    return ESTIMATORS[name][0]


@dataclasses.dataclass
class TrainingConfig:
    """Convenience wrapper for model training parameters and file info

    Attrs:
        model_type: sklearn model type or keras model class to initialize
        input_variables: variables used as features
        output_variables: variables to predict
        additional_variables: list of needed variables which are not inputs
            or outputs (e.g. pressure thickness if needed for scaling)
        hyperparameters: arguments to pass to model class at initialization
            time
        random_seed: value to use to initialize randomness
        model_path: output location for final model
        batch_function: name of function from `fv3fit.batches` to use for
            loading batched data
        batch_kwargs: keyword arguments to pass to batch function
        data_path: location of training data to be loaded by batch function
        scaler_type: scaler to use for training
        scaler_kwargs: keyword arguments to pass to scaler initialization
        validation_timesteps: timestamps to use as validation samples
        save_model_checkpoints: whether to save a copy of the model at
            each epoch
        timesteps_source: one of "timesteps_file",
            "sampled_outside_input_config", "input_config", "all_mapper_times"
    """

    model_type: str
    input_variables: List[str]
    output_variables: List[str]
    additional_variables: List[str] = dataclasses.field(default_factory=list)
    hyperparameters: dict = dataclasses.field(default_factory=dict)
    random_seed: Union[float, int] = 0
    model_path: str = ""

    @classmethod
    def from_dict(cls, kwargs) -> "TrainingConfig":
        if cls is TrainingConfig:
            subclass = get_config_class(kwargs["model_type"])
            result = subclass.from_dict(kwargs)
        else:
            result = cls(**kwargs)
        return result


ESTIMATORS: Dict[str, Tuple[Type[Estimator], Type[TrainingConfig]]] = {}


def get_config_class(model_type: str) -> Type[TrainingConfig]:
    if model_type in ESTIMATORS:
        _, subclass = ESTIMATORS[model_type]
    else:
        raise ValueError(f"unknown model_type {model_type}")
    return subclass


def get_estimator_class(model_type: str) -> Type[TrainingConfig]:
    if model_type in ESTIMATORS:
        estimator_class, _ = ESTIMATORS[model_type]
    else:
        raise ValueError(f"unknown model_type {model_type}")
    return estimator_class


def register_estimator(name: str, config_class: type):
    """
    Returns a decorator that will register the given class as a keras training
    class, which can be used in training configuration.
    """

    def decorator(cls):
        ESTIMATORS[name] = (cls, config_class)
        return cls

    return decorator


@dataclasses.dataclass
class DataConfig:
    """Convenience wrapper for model training data

    Attrs:
        variables: names of variables to include in dataset
        data_path: location of training data to be loaded by batch function
        batch_function: name of function from `fv3fit.batches` to use for
            loading batched data
        batch_kwargs: keyword arguments to pass to batch function
    """

    variables: List[str]
    data_path: str
    batch_function: str
    batch_kwargs: dict


# TODO: move this class to where the Dense training is defined when config.py
# no longer depends on it (i.e. when _ModelTrainingConfig is deleted)
@dataclasses.dataclass
class DenseTrainingConfig(TrainingConfig):
    """
    Attrs:
        model_type: sklearn model type or keras model class to initialize
        input_variables: variables used as features
        output_variables: variables to predict
        additional_variables: list of needed variables which are not inputs
            or outputs (e.g. pressure thickness if needed for scaling)
        hyperparameters: arguments to pass to model class at initialization
            time
        random_seed: value to use to initialize randomness
        model_path: output location for final model
        save_model_checkpoints: whether to save a copy of the model at
            each epoch
    """

    save_model_checkpoints: bool = False


@dataclasses.dataclass
class SklearnTrainingConfig(TrainingConfig):
    """
    Attrs:
        model_type: sklearn model type or keras model class to initialize
        input_variables: variables used as features
        output_variables: variables to predict
        additional_variables: list of needed variables which are not inputs
            or outputs (e.g. pressure thickness if needed for scaling)
        hyperparameters: arguments to pass to model class at initialization
            time
        random_seed: value to use to initialize randomness
        model_path: output location for final model
        scaler_type: scaler to use for training
        scaler_kwargs: keyword arguments to pass to scaler initialization
    """

    scaler_type: str = "standard"
    scaler_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # TODO: turn this into an exception if DELP is not present
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)


@dataclasses.dataclass
class _ModelTrainingConfig:
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
    def load(cls, path: str) -> "_ModelTrainingConfig":
        with fsspec.open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return _ModelTrainingConfig(**config_dict)


def get_model(config: TrainingConfig):
    model_class, _ = ESTIMATORS[config.model_type]
    return model_class(**config.hyperparameters)


def legacy_config_to_new_config(legacy_config: _ModelTrainingConfig) -> TrainingConfig:
    config_dict = dataclasses.asdict(legacy_config)
    config_class = ESTIMATORS[legacy_config.model_type][1]
    if config_class is SklearnTrainingConfig:
        keys = [
            "model_type",
            "hyperparameters",
            "input_variables",
            "output_variables",
            "additional_variables",
            "random_seed",
            "model_path",
            "scaler_type",
            "scaler_kwargs",
        ]
    elif config_class is DenseTrainingConfig:
        keys = [
            "model_type",
            "hyperparameters",
            "input_variables",
            "output_variables",
            "additional variables",
            "random_seed",
            "model_path",
            "save_model_checkpoints",
        ]
        fit_kwargs = legacy_config.hyperparameters.pop("fit_kwargs", {})
        fit_kwargs["validation_dataset"] = validation_dataset(legacy_config)
        legacy_config.hyperparameters["fit_kwargs"] = fit_kwargs
    else:
        raise NotImplementedError(f"unknown model type {legacy_config.model_type}")
    training_config = TrainingConfig.from_dict(
        {key: config_dict[key] for key in keys if key in config_dict}
    )
    return training_config


def load_configs(
    config_path: str,
    data_path: str,
    output_data_path: str,
    timesteps_file=None,
    validation_timesteps_file=None,
) -> Tuple[_ModelTrainingConfig, TrainingConfig, DataConfig, Optional[DataConfig]]:
    """Load training configuration information from a legacy yaml config path.

    Dumps the legacy configuration class to the output_data_path.
    """
    # TODO: remove output_data_path argument, we need it here at the moment
    # to dump legacy_config before it gets a Dataset attached to it,
    # for backwards compatibility
    # we shouldn't need this when validation_dataset is in its own data config
    # and not attached to fit_kwargs
    legacy_config = _ModelTrainingConfig.load(config_path)
    legacy_config.data_path = data_path
    legacy_config.dump(output_data_path)
    training_config = legacy_config_to_new_config(legacy_config)
    config_dict = dataclasses.asdict(legacy_config)

    variables = (
        config_dict["input_variables"]
        + config_dict["output_variables"]
        + config_dict.get("additional_variables", [])
    )
    data_path = config_dict["data_path"]
    batch_function = config_dict["batch_function"]
    batch_kwargs = config_dict["batch_kwargs"]

    train_batch_kwargs = {**batch_kwargs}
    if timesteps_file is not None:
        with open(timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_batch_kwargs["timesteps"] = timesteps
    train_data_config = DataConfig(
        variables=variables,
        data_path=data_path,
        batch_function=batch_function,
        batch_kwargs=train_batch_kwargs,
    )

    if validation_timesteps_file is not None:
        validation_batch_kwargs = {**batch_kwargs}
        with open(validation_timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        validation_batch_kwargs["timesteps"] = timesteps
        validation_data_config: Optional[DataConfig] = DataConfig(
            variables=variables,
            data_path=data_path,
            batch_function=batch_function,
            batch_kwargs=validation_batch_kwargs,
        )
    else:
        validation_data_config = None

    return legacy_config, training_config, train_data_config, validation_data_config


# TODO: this should be made to work regardless of whether we're using
# keras or sklearn models, find a way to delete this code entirely and
# use the validation DataConfig instead.


def check_validation_train_overlap(
    train: Sequence[str], validate: Sequence[str]
) -> None:
    overlap = set(train) & set(validate)
    if overlap:
        raise ValueError(
            f"Timestep(s) {overlap} are in both train and validation sets."
        )


def validation_timesteps_config(train_config):
    val_config = legacy_config_to_data_config(train_config)
    assert not isinstance(val_config.data_path, list)
    val_config.batch_kwargs["timesteps"] = train_config.validation_timesteps
    val_config.batch_kwargs["timesteps_per_batch"] = len(
        train_config.validation_timesteps  # type: ignore
    )
    return val_config


# TODO: refactor all tests and code using this to create DataConfig from the beginning
# and delete this helper routine
def legacy_config_to_data_config(legacy_config):
    return DataConfig(
        variables=legacy_config.input_variables
        + legacy_config.output_variables
        + (legacy_config.additional_variables or []),
        data_path=legacy_config.data_path,
        batch_function=legacy_config.batch_function,
        batch_kwargs=legacy_config.batch_kwargs,
    )


def validation_dataset(train_config: _ModelTrainingConfig,) -> Optional[xr.Dataset]:
    if len(train_config.validation_timesteps) > 0:
        check_validation_train_overlap(
            train_config.batch_kwargs["timesteps"], train_config.validation_timesteps,
        )
        validation_config = validation_timesteps_config(train_config)
        # validation config puts all data in one batch
        validation_dataset_sequence = load_data_sequence(validation_config)
        if len(validation_dataset_sequence) > 1:
            raise ValueError(
                "Something went wrong! "
                "All validation data should be concatenated into a single batch. There "
                f"are {len(validation_dataset_sequence)} batches in the sequence."
            )
        return validation_dataset_sequence[0]
    else:
        validation_dataset = None
    return validation_dataset


def load_training_config(model_path: str) -> _ModelTrainingConfig:
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
    return _ModelTrainingConfig.load(config_path)


# TODO: move this back into its own module once it has been decoupled
# from the config code by deleting `validation_dataset`
def load_data_sequence(config: DataConfig) -> batches.BaseSequence[xr.Dataset]:
    """
    Args:
        config: data configuration

    Returns:
        Sequence of datasets according to configuration
    """
    batch_function = getattr(batches, config.batch_function)
    ds_batches = batch_function(
        config.data_path, list(config.variables), **config.batch_kwargs,
    )
    return ds_batches

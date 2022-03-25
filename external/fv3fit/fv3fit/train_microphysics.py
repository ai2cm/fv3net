import argparse
import dacite
from enum import Enum
from dataclasses import asdict, dataclass, field
import fsspec
import json
import logging
import numpy as np
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union
from fv3fit._shared.config import register_training_function

import tensorflow as tf
import yaml

from fv3fit import set_random_seed
from fv3fit._shared import put_dir
from fv3fit._shared.config import (
    OptimizerConfig,
    get_arg_updated_config_dict,
    to_nested_dict,
)
from fv3fit.emulation.layers.normalization2 import MeanMethod, StdDevMethod
from fv3fit.keras._models.shared.pure_keras import PureKerasDictPredictor
from fv3fit.keras.jacobian import compute_jacobians, nondimensionalize_jacobians

from fv3fit.emulation.transforms.factories import ConditionallyScaled
from fv3fit.emulation.types import LossFunction, TensorDict
from fv3fit.emulation import models, train, ModelCheckpointCallback
from fv3fit.emulation.data import TransformConfig, nc_dir_to_tfdataset
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.keras import save_model
from fv3fit.emulation.losses import CustomLoss
from fv3fit.emulation.models import transform_model
from fv3fit.emulation.transforms import (
    ComposedTransformFactory,
    Difference,
    TensorTransform,
    TransformedVariableConfig,
)
from fv3fit.emulation.layers.normalization import standard_deviation_all_features
from fv3fit.wandb import (
    WandBConfig,
    store_model_artifact,
    plot_all_output_sensitivities,
)

logger = logging.getLogger(__name__)


def _asdict_with_enum(obj):
    """Recursively turn a dataclass obj into a dictionary handling any Enums
    """

    def _generate(x):
        for key, val in x:
            if isinstance(val, Enum):
                yield key, val.value
            else:
                yield key, val

    def dict_factory(x):
        return dict(_generate(x))

    return asdict(obj, dict_factory=dict_factory)


def load_config_yaml(path: str) -> Dict[str, Any]:
    """
    Load yaml from local/remote location
    """

    with fsspec.open(path, "r") as f:
        d = yaml.safe_load(f)

    return d


@dataclass
class TransformedParameters:
    """
    Configuration for training a microphysics emulator

    Args:
        out_url:  where to save checkpoints
        transform: Data preprocessing TransformConfig
        tensor_transform: specification of differerentiable tensorflow
            transformations to apply before and after data is passed to models and
            losses.
        model: MicrophysicsConfig used to build the keras model
        nfiles: Number of files to use from train_url
        nfiles_valid: Number of files to use from test_url
        use_wandb: Enable wandb logging of training, requires that wandb is installed
            and initialized
        wandb: WandBConfig to set up the wandb logged run
        loss:  Configuration of the keras loss to prepare and use for training
        epochs: Number of training epochs
        batch_size: batch size applied to tf datasets during training
        valid_freq: How often to score validation data (in epochs)
        verbose: Verbosity of keras fit output
        shuffle_buffer_size: How many samples to keep in the keras shuffle buffer
            during training
        checkpoint_model: if true, save a checkpoint after each epoch
        log_level: what logging level to use
    """

    tensor_transform: List[
        Union[TransformedVariableConfig, ConditionallyScaled, Difference]
    ] = field(default_factory=list)
    model: Optional[models.MicrophysicsConfig] = None
    conservative_model: Optional[models.ConservativeWaterConfig] = None
    loss: CustomLoss = field(default_factory=CustomLoss)
    epochs: int = 1
    batch_size: int = 128
    valid_freq: int = 5
    verbose: int = 2
    shuffle_buffer_size: Optional[int] = 13824
    # only model checkpoints are saved at out_url, but need to keep these name
    # for backwards compatibility
    checkpoint_model: bool = True
    out_url: str = ""
    # ideally will refactor these out, but need to insert the callback somehow
    use_wandb: bool = True
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @property
    def transform_factory(self) -> ComposedTransformFactory:
        return ComposedTransformFactory(self.tensor_transform)

    def build_transform(self, sample: TensorDict) -> TensorTransform:
        return self.transform_factory.build(sample)

    @property
    def _model(
        self,
    ) -> Union[
        models.MicrophysicsConfig, models.ConservativeWaterConfig,
    ]:
        if self.model:
            return self.model
        elif self.conservative_model:
            return self.conservative_model
        else:
            raise ValueError(
                "Neither .model, .conservative_model, nor .transformed_model provided."
            )

    def build_model(
        self, data: Mapping[str, tf.Tensor], transform: TensorTransform
    ) -> tf.keras.Model:
        inputs = {
            name: tf.keras.Input(data[name].shape[1:], name=name)
            for name in self.input_variables
        }
        inner_model = self._model.build(transform.forward(data))
        return transform_model(inner_model, transform, inputs)

    def build_loss(
        self, data: Mapping[str, tf.Tensor], transform: TensorTransform
    ) -> LossFunction:
        return self.loss.build(transform.forward(data))

    @property
    def input_variables(self) -> Sequence:
        return list(
            self.transform_factory.backward_names(set(self._model.input_variables))
        )

    @property
    def model_variables(self) -> Set[str]:
        return self.transform_factory.backward_names(
            set(self._model.input_variables) | set(self._model.output_variables)
        )


# Temporarily subclass from the hyperparameters object for backwards compatibility
# we can delete this class once usage has switched to fv3fit.train
@dataclass
class TrainConfig(TransformedParameters):
    """
    Configuration for training a microphysics emulator

    Args:
        train_url: Path to training netcdfs (already in [sample x feature] format)
        test_url: Path to validation netcdfs (already in [sample x feature] format)
        out_url:  Where to store the trained model, history, and configuration
        transform: Data preprocessing TransformConfig
        tensor_transform: specification of differerentiable tensorflow
            transformations to apply before and after data is passed to models and
            losses.
        model: MicrophysicsConfig used to build the keras model
        nfiles: Number of files to use from train_url
        nfiles_valid: Number of files to use from test_url
        use_wandb: Enable wandb logging of training, requires that wandb is installed
            and initialized
        wandb: WandBConfig to set up the wandb logged run
        loss:  Configuration of the keras loss to prepare and use for training
        epochs: Number of training epochs
        batch_size: batch size applied to tf datasets during training
        valid_freq: How often to score validation data (in epochs)
        verbose: Verbosity of keras fit output
        shuffle_buffer_size: How many samples to keep in the keras shuffle buffer
            during training
        checkpoint_model: if true, save a checkpoint after each epoch
        log_level: what logging level to use
    """

    train_url: str = ""
    test_url: str = ""
    transform: TransformConfig = field(default_factory=TransformConfig)
    tensor_transform: List[
        Union[TransformedVariableConfig, ConditionallyScaled, Difference]
    ] = field(default_factory=list)
    model: Optional[models.MicrophysicsConfig] = None
    conservative_model: Optional[models.ConservativeWaterConfig] = None
    nfiles: Optional[int] = None
    nfiles_valid: Optional[int] = None
    loss: CustomLoss = field(default_factory=CustomLoss)
    epochs: int = 1
    batch_size: int = 128
    valid_freq: int = 5
    verbose: int = 2
    shuffle_buffer_size: Optional[int] = 13824
    checkpoint_model: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Standard init from nested dictionary"""
        # casting necessary for 'from_args' which all come in as string
        # TODO: should this be just a json parsed??
        config = dacite.Config(
            strict=True, cast=[bool, str, int, float, StdDevMethod, MeanMethod]
        )
        return dacite.from_dict(cls, d, config=config)

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """
        Init from a dictionary flattened in the style of wandb configs
        where all nested mapping keys are flattened to the top level
        by joining with a '.'

        E.g.:
        {
            "test_url": "gs://bucket/path/to/blobs",
            "model.input_variables": ["var1", "var2"],
            "model.architecture.name": "rnn",
            ...
        }
        """
        d = to_nested_dict(d)
        return cls.from_dict(d)

    @classmethod
    def from_yaml_path(cls, path: str) -> "TrainConfig":
        """Init from path to yaml file"""
        d = load_config_yaml(path)
        return cls.from_dict(d)

    @classmethod
    def from_args(cls, args: Optional[Sequence[str]] = None):
        """
        Init from commandline arguments (or provided arguments).  If no args
        are provided, uses sys.argv to parse.

        Note: A current limitation of this init style is that we cannot provide
        arbitrary arguments to the parser.  Therefore, value being updated should
        either be a member of the default config or the file specified by
        --config-path

        Args:
            args: A list of arguments to be parsed.  If not provided, uses
                sys.argv

                Requires "--config-path", use "--config-path default" to use
                default configuration

                Note: arguments should be in the flat style used by wandb where all
                nested mappings are at the top level with '.' joined keys. E.g.,
                "--model.architecture.name rnn"
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config-path",
            required=True,
            help="Path to training config yaml. Use '--config-path default'"
            " to run with a default configuration.",
        )

        path_arg, unknown_args = parser.parse_known_args(args=args)

        if path_arg.config_path == "default":
            config = get_default_config()
        else:
            config = cls.from_yaml_path(path_arg.config_path)

        if unknown_args:
            updated = get_arg_updated_config_dict(
                unknown_args, _asdict_with_enum(config)
            )
            config = cls.from_dict(updated)

        return config

    def to_yaml(self) -> str:
        return yaml.safe_dump(_asdict_with_enum(self))

    def open_dataset(
        self, url: str, nfiles: Optional[int], required_variables: Set[str],
    ) -> tf.data.Dataset:
        nc_open_fn = self.transform.get_pipeline(required_variables)
        return nc_dir_to_tfdataset(
            url,
            nc_open_fn,
            nfiles=nfiles,
            shuffle=True,
            random_state=np.random.RandomState(0),
        )


def save_jacobians(std_jacobians, dir_, filename="jacobians.npz"):
    with put_dir(dir_) as tmpdir:
        dumpable = {
            f"{out_name}/{in_name}": data
            for out_name, sensitivities in std_jacobians.items()
            for in_name, data in sensitivities.items()
        }
        np.savez(os.path.join(tmpdir, filename), **dumpable)


@register_training_function("transformed", TransformedParameters)
def train_function(
    config: TransformedParameters, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset,
) -> PureKerasDictPredictor:
    return _train_function_unbatched(config, train_ds.unbatch(), test_ds.unbatch())


def _train_function_unbatched(
    config: TransformedParameters, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset,
) -> PureKerasDictPredictor:
    # callbacks that are always active
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]

    if config.use_wandb:
        config.wandb.init(config=_asdict_with_enum(config))
        callbacks.append(config.wandb.get_callback())

    if config.shuffle_buffer_size is not None:
        train_ds = train_ds.shuffle(config.shuffle_buffer_size)

    train_set = next(iter(train_ds.batch(50_000)))

    transform = config.build_transform(train_set)

    train_ds = train_ds.map(transform.forward)
    test_ds = test_ds.map(transform.forward)

    model = config.build_model(train_set, transform)

    if config.checkpoint_model:
        callbacks.append(
            ModelCheckpointCallback(
                filepath=os.path.join(
                    config.out_url, "checkpoints", "epoch.{epoch:03d}.tf"
                )
            )
        )

    train_ds_batched = train_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds_batched = test_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    history = train(
        model,
        train_ds_batched,
        config.build_loss(train_set, transform),
        optimizer=config.loss.optimizer.instance,
        epochs=config.epochs,
        validation_data=test_ds_batched,
        validation_freq=config.valid_freq,
        verbose=config.verbose,
        callbacks=callbacks,
    )

    return PureKerasDictPredictor(
        model, passthrough=(model, transform, history, train_set)
    )


def main(config: TrainConfig, seed: int = 0):
    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    train_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )
    test_ds = config.open_dataset(
        config.test_url, config.nfiles_valid, config.model_variables
    )

    predictor = train_function(config, train_ds, test_ds)
    model, transform, history, train_set = predictor.passthrough  # type: ignore

    logger.debug("Training complete")

    with put_dir(config.out_url) as tmpdir:

        with open(os.path.join(tmpdir, "history.json"), "w") as f:
            json.dump(history.params, f)

        with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
            f.write(config.to_yaml())

        local_model_path = save_model(model, tmpdir)

        if config.use_wandb:
            store_model_artifact(local_model_path, name=config._model.name)

    # Jacobians after model storing in case of "out of memory" errors
    sample = transform.forward(train_set)
    jacobians = compute_jacobians(model, sample, config.input_variables)
    std_factors = {
        name: np.array(float(standard_deviation_all_features(data)))
        for name, data in sample.items()
    }
    std_jacobians = nondimensionalize_jacobians(jacobians, std_factors)

    save_jacobians(std_jacobians, config.out_url, "jacobians.npz")
    if config.use_wandb:
        plot_all_output_sensitivities(std_jacobians)


def get_default_config():

    input_vars = [
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "pressure_thickness_of_atmospheric_layer",
    ]

    model_config = models.MicrophysicsConfig(
        input_variables=input_vars,
        direct_out_variables=[
            "cloud_water_mixing_ratio_after_precpd",
            "total_precipitation",
        ],
        residual_out_variables=dict(
            air_temperature_after_precpd="air_temperature_input",
            specific_humidity_after_precpd="specific_humidity_input",
        ),
        architecture=ArchitectureConfig("linear"),
        selection_map=dict(
            air_temperature_input=SliceConfig(stop=-10),
            specific_humidity_input=SliceConfig(stop=-10),
            cloud_water_mixing_ratio_input=SliceConfig(stop=-10),
            pressure_thickness_of_atmospheric_layer=SliceConfig(stop=-10),
        ),
        tendency_outputs=dict(
            air_temperature_after_precpd="tendency_of_air_temperature_due_to_microphysics",  # noqa E501
            specific_humidity_after_precpd="tendency_of_specific_humidity_due_to_microphysics",  # noqa E501
        ),
    )

    transform = TransformConfig()

    loss = CustomLoss(
        optimizer=OptimizerConfig(name="Adam", kwargs=dict(learning_rate=1e-4)),
        loss_variables=[
            "air_temperature_after_precpd",
            "specific_humidity_after_precpd",
            "cloud_water_mixing_ratio_after_precpd",
            "total_precipitation",
        ],
        weights=dict(
            air_temperature_after_precpd=0.5e5,
            specific_humidity_after_precpd=0.5e5,
            cloud_water_mixing_ratio_after_precpd=1.0,
            total_precipitation=0.04,
        ),
        metric_variables=[
            "tendency_of_air_temperature_due_to_microphysics",
            "tendency_of_specific_humidity_due_to_microphysics",
            "tendency_of_cloud_water_mixing_ratio_due_to_microphysics",
        ],
    )

    config = TrainConfig(
        train_url="gs://vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/train",  # noqa E501
        test_url="gs://vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/test",  # noqa E501
        out_url="gs://vcm-ml-scratch/andrep/test-train-emulation",
        model=model_config,
        transform=transform,
        loss=loss,
        nfiles=80,
        nfiles_valid=80,
        valid_freq=1,
        epochs=4,
        wandb=WandBConfig(job_type="training"),
    )

    return config


if __name__ == "__main__":

    config = TrainConfig.from_args()
    main(config)

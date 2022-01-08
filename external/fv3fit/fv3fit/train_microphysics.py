import argparse
import dacite
from dataclasses import asdict, dataclass, field
import fsspec
import json
import logging
import numpy as np
import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import tensorflow as tf
import yaml

from fv3fit import set_random_seed
from fv3fit._shared import put_dir
from fv3fit._shared.config import (
    OptimizerConfig,
    get_arg_updated_config_dict,
    to_nested_dict,
)
from fv3fit.emulation.types import LossFunction
from fv3fit.emulation import models, train, ModelCheckpointCallback
from fv3fit.emulation.data import TransformConfig, nc_dir_to_tf_dataset
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.jacobian import compute_standardized_jacobians
from fv3fit.emulation.keras import save_model
from fv3fit.emulation.losses import CustomLoss
from fv3fit.emulation.transforms import (
    PerVariableTransform,
    TensorTransform,
    TransformedVariableConfig,
)
import xarray
from fv3fit.wandb import (
    WandBConfig,
    store_model_artifact,
    plot_all_output_sensitivities,
)

logger = logging.getLogger(__name__)


def load_config_yaml(path: str) -> Dict[str, Any]:
    """
    Load yaml from local/remote location
    """

    with fsspec.open(path, "r") as f:
        d = yaml.safe_load(f)

    return d


@dataclass
class TrainConfig:
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
        cache: Use a cache for training/testing batches. Speeds up training for
            I/O bound architectures.  Always disabled for rnn-v1 architectures.
    """

    train_url: str
    test_url: str
    out_url: str
    transform: TransformConfig = field(default_factory=TransformConfig)
    tensor_transform: List[TransformedVariableConfig] = field(default_factory=list)
    model: Optional[models.MicrophysicsConfig] = None
    conservative_model: Optional[models.ConservativeWaterConfig] = None
    transformed_model: Optional[models.TransformedModelConfig] = None
    nfiles: Optional[int] = None
    nfiles_valid: Optional[int] = None
    use_wandb: bool = True
    wandb: WandBConfig = field(default_factory=WandBConfig)
    loss: CustomLoss = field(default_factory=CustomLoss)
    epochs: int = 1
    batch_size: int = 128
    valid_freq: int = 5
    verbose: int = 2
    shuffle_buffer_size: Optional[int] = 100_000
    checkpoint_model: bool = True
    log_level: str = "INFO"
    cache: bool = True

    def get_transform(self) -> TensorTransform:
        return PerVariableTransform(self.tensor_transform)

    @property
    def _model(
        self,
    ) -> Union[
        models.MicrophysicsConfig,
        models.ConservativeWaterConfig,
        models.TransformedModelConfig,
    ]:
        if self.model:
            return self.model
        elif self.conservative_model:
            return self.conservative_model
        elif self.transformed_model:
            return self.transformed_model
        else:
            raise ValueError(
                "Neither .model, .conservative_model, nor .transformed_model provided."
            )

    def build_model(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        return self._model.build(data, self.get_transform())

    def build_loss(self, data: Mapping[str, tf.Tensor]) -> LossFunction:
        return self.loss.build(data, self.get_transform())

    @property
    def input_variables(self) -> Sequence:
        return list(
            self.get_transform().backward_names(set(self._model.input_variables))
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Standard init from nested dictionary"""
        # casting necessary for 'from_args' which all come in as string
        # TODO: should this be just a json parsed??
        config = dacite.Config(strict=True, cast=[bool, str, int, float])
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
            updated = get_arg_updated_config_dict(unknown_args, asdict(config))
            config = cls.from_dict(updated)

        return config

    def get_dataset_convertor(
        self,
    ) -> Callable[[xarray.Dataset], Mapping[str, tf.Tensor]]:
        model_variables = set(self._model.input_variables) | set(
            self._model.output_variables
        )
        required_variables = self.get_transform().backward_names(model_variables)
        return self.transform.get_pipeline(required_variables)

    def __post_init__(self) -> None:
        if (
            self.model is not None
            and "rnn-v1" in self.model.architecture.name
            and self.cache
        ):
            logger.warn("Caching disabled for rnn-v1 architectures due to memory leak")
            self.cache = False


def save_jacobians(std_jacobians, dir_, filename="jacobians.npz"):
    with put_dir(dir_) as tmpdir:
        dumpable = {
            f"{out_name}/{in_name}": data
            for out_name, sensitivities in std_jacobians.items()
            for in_name, data in sensitivities.items()
        }
        np.savez(os.path.join(tmpdir, filename), **dumpable)


def main(config: TrainConfig, seed: int = 0):
    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    callbacks = []
    if config.use_wandb:
        config.wandb.init(config=asdict(config))
        callbacks.append(config.wandb.get_callback())

    train_ds = nc_dir_to_tf_dataset(
        config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
    )
    test_ds = nc_dir_to_tf_dataset(
        config.test_url, config.get_dataset_convertor(), nfiles=config.nfiles_valid
    )

    train_set = next(iter(train_ds.shuffle(100_000).batch(50_000)))

    model = config.build_model(train_set)

    if config.shuffle_buffer_size is not None:
        train_ds = train_ds.shuffle(config.shuffle_buffer_size)

    if config.checkpoint_model:
        callbacks.append(
            ModelCheckpointCallback(
                filepath=os.path.join(
                    config.out_url, "checkpoints", "epoch.{epoch:03d}.tf"
                )
            )
        )

    with tempfile.TemporaryDirectory() as train_temp:
        with tempfile.TemporaryDirectory() as test_temp:

            train_ds_batched = train_ds.batch(config.batch_size)
            test_ds_batched = test_ds.batch(config.batch_size)

            if config.cache:
                train_ds_batched = train_ds_batched.cache(train_temp)
                test_ds_batched = test_ds_batched.cache(test_temp)

            history = train(
                model,
                train_ds_batched,
                config.build_loss(train_set),
                optimizer=config.loss.optimizer.instance,
                epochs=config.epochs,
                validation_data=test_ds_batched,
                validation_freq=config.valid_freq,
                verbose=config.verbose,
                callbacks=callbacks,
            )

    logger.debug("Training complete")

    with put_dir(config.out_url) as tmpdir:

        with open(os.path.join(tmpdir, "history.json"), "w") as f:
            json.dump(history.params, f)

        with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
            f.write(yaml.safe_dump(asdict(config)))

        local_model_path = save_model(model, tmpdir)

        if config.use_wandb:
            store_model_artifact(local_model_path, name=config._model.name)

    # Jacobians after model storing in case of "out of memory" errors
    std_jacobians = compute_standardized_jacobians(
        model, config.get_transform().forward(train_set), config.input_variables
    )
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

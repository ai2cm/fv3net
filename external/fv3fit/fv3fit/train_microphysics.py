import argparse
import json
import logging
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import dacite
import fsspec
import numpy as np
import tensorflow as tf
import yaml
from fv3fit import set_random_seed
from fv3fit._shared import put_dir
from fv3fit._shared.config import (
    OptimizerConfig,
    get_arg_updated_config_dict,
    to_nested_dict,
)
from fv3fit.emulation import models, Trainer, ModelCheckpointCallback
from fv3fit.emulation.data import TransformConfig, nc_dir_to_tf_dataset
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.keras import CustomLoss, StandardLoss, save_model, score_model
from fv3fit.wandb import (
    WandBConfig,
    log_profile_plots,
    log_to_table,
    store_model_artifact,
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

    train_url: str
    test_url: str
    out_url: str
    transform: TransformConfig = field(default_factory=TransformConfig)
    model: Optional[models.MicrophysicsConfig] = None
    conservative_model: Optional[models.ConservativeWaterConfig] = None
    nfiles: Optional[int] = None
    nfiles_valid: Optional[int] = None
    use_wandb: bool = True
    wandb: WandBConfig = field(default_factory=WandBConfig)
    loss: Union[StandardLoss, CustomLoss] = field(default_factory=StandardLoss)
    epochs: int = 1
    batch_size: int = 128
    valid_freq: int = 5
    verbose: int = 2
    shuffle_buffer_size: Optional[int] = 100_000
    checkpoint_model: bool = True
    log_level: str = "INFO"

    @property
    def _model(
        self,
    ) -> Union[models.MicrophysicsConfig, models.ConservativeWaterConfig]:
        if self.model:
            if self.conservative_model:
                warnings.warn(
                    UserWarning(
                        ".conservative_model included in the configuration, "
                        "but will not be used."
                    )
                )

            return self.model
        elif self.conservative_model:
            return self.conservative_model
        else:
            raise ValueError("Neither .model or .conservative_model provided.")

    def build(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        return self._model.build(data)

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

    def __post_init__(self) -> None:
        required_variables = set(self._model.input_variables) | set(
            self._model.output_variables
        )
        self.transform.variables = list(required_variables)


def main(config: TrainConfig, seed: int = 0):
    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    callbacks = []
    if config.use_wandb:
        config.wandb.init(config=asdict(config))
        callbacks.append(config.wandb.get_callback())

    train_ds = nc_dir_to_tf_dataset(
        config.train_url, config.transform, nfiles=config.nfiles
    )
    test_ds = nc_dir_to_tf_dataset(
        config.test_url, config.transform, nfiles=config.nfiles_valid
    )

    train_set = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    test_set = next(iter(test_ds.shuffle(160_000).batch(80_000)))

    model = config.build(train_set)
    output_names = set(model(train_set))

    if config.shuffle_buffer_size is not None:
        train_ds = train_ds.shuffle(config.shuffle_buffer_size)

    def split_in_out(m):
        return (
            {key: m[key] for key in model.input_names if key in m},
            {key: m[key] for key in output_names if key in m},
        )

    trainer = Trainer(model)
    config.loss.prepare(output_samples=train_set)
    config.loss.compile(trainer)

    if config.checkpoint_model:
        callbacks.append(
            ModelCheckpointCallback(
                filepath=os.path.join(
                    config.out_url, "checkpoints", "epoch.{epoch:03d}.tf"
                )
            )
        )
    train_ds = train_ds.map(split_in_out)
    test_ds = test_ds.map(split_in_out)

    with tempfile.TemporaryDirectory() as train_temp:
        with tempfile.TemporaryDirectory() as test_temp:

            train_ds_cached = train_ds.batch(config.batch_size).cache(train_temp)
            test_ds_cached = test_ds.batch(config.batch_size).cache(test_temp)

            history = trainer.fit(
                train_ds_cached,
                epochs=config.epochs,
                validation_data=test_ds_cached,
                validation_freq=config.valid_freq,
                verbose=config.verbose,
                callbacks=callbacks,
            )
    logger.debug("Training complete")
    train_scores, train_profiles = score_model(model, train_set)
    test_scores, test_profiles = score_model(model, test_set)
    logger.debug("Scoring Complete")

    if config.use_wandb:
        pred_sample = model.predict(test_set)
        log_profile_plots(test_set, pred_sample)

        # add level for dataframe index, assumes equivalent feature dims
        sample_profile = next(iter(train_profiles.values()))
        train_profiles["level"] = np.arange(len(sample_profile))
        test_profiles["level"] = np.arange(len(sample_profile))

        log_to_table("score/train", train_scores, index=[config.wandb.job.name])
        log_to_table("score/test", test_scores, index=[config.wandb.job.name])
        log_to_table("profiles/train", train_profiles)
        log_to_table("profiles/test", test_profiles)

    with put_dir(config.out_url) as tmpdir:
        # TODO: need to convert ot np.float to serialize
        with open(os.path.join(tmpdir, "scores.json"), "w") as f:
            json.dump({"train": train_scores, "test": test_scores}, f)

        with open(os.path.join(tmpdir, "history.json"), "w") as f:
            json.dump(history.params, f)

        with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
            f.write(yaml.safe_dump(asdict(config)))

        local_model_path = save_model(model, tmpdir)

        if config.use_wandb:
            store_model_artifact(local_model_path, name=config._model.name)


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
            "cloud_water_mixing_ratio_output",
            "total_precipitation",
        ],
        residual_out_variables=dict(
            air_temperature_output="air_temperature_input",
            specific_humidity_output="specific_humidity_input",
        ),
        architecture=ArchitectureConfig("linear"),
        selection_map=dict(
            air_temperature_input=SliceConfig(stop=-10),
            specific_humidity_input=SliceConfig(stop=-10),
            cloud_water_mixing_ratio_input=SliceConfig(stop=-10),
            pressure_thickness_of_atmospheric_layer=SliceConfig(stop=-10),
        ),
        tendency_outputs=dict(
            air_temperature_output="tendency_of_air_temperature_due_to_microphysics",  # noqa E501
            specific_humidity_output="tendency_of_specific_humidity_due_to_microphysics",  # noqa E501
        ),
    )

    transform = TransformConfig()

    loss = CustomLoss(
        optimizer=OptimizerConfig(name="Adam", kwargs=dict(learning_rate=1e-4)),
        loss_variables=[
            "air_temperature_output",
            "specific_humidity_output",
            "cloud_water_mixing_ratio_output",
            "total_precipitation",
        ],
        weights=dict(
            air_temperature_output=0.5e5,
            specific_humidity_output=0.5e5,
            cloud_water_mixing_ratio_output=1.0,
            total_precipitation=0.04,
        ),
        metric_variables=[
            "tendency_of_air_temperature_due_to_microphysics",
            "tendency_of_specific_humidity_due_to_microphysics",
            "tendency_of_cloud_water_mixing_ratio_due_to_microphysics",
        ],
    )

    config = TrainConfig(
        train_url="gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/training_netcdfs",  # noqa E501
        test_url="gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/validation_netcdfs",  # noqa E501
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

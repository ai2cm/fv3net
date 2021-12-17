import argparse
import dacite
from dataclasses import asdict, dataclass, field
import fsspec
import json
import logging
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import tensorflow as tf
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import wandb
import warnings
import yaml

from fv3fit import set_random_seed
from fv3fit._shared import put_dir
from fv3fit._shared.config import (
    OptimizerConfig,
    get_arg_updated_config_dict,
    to_nested_dict,
)
from fv3fit.emulation import models, train, ModelCheckpointCallback
from fv3fit.emulation.data import TransformConfig, nc_dir_to_tf_dataset
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.keras import CustomLoss, StandardLoss, save_model
from fv3fit.wandb import (
    WandBConfig,
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
        cache: Use a cache for training/testing batches. Speeds up training for
            I/O bound architectures.  Always disabled for rnn-v1 architectures.
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
    cache: bool = True

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

        if (
            self.model is not None
            and "rnn-v1" in self.model.architecture.name
            and self.cache
        ):
            logger.warn("Caching disabled for rnn-v1 architectures due to memory leak")
            self.cache = False


def get_model_output_sensitivities(model: tf.keras.Model, sample: Mapping[str, tf.Tensor]):
    """
    Generate sensitivity jacabion for each output relative to a mean
    value for each input
    """

    avg_profiles = {
        name: tf.reduce_mean(data, axis=0, keepdims=True)
        for name, data in sample.items()
    }

    # normalize factors so sensitivities are comparable but still
    # preserve level-relative magnitudes
    normalize_factors = {}
    for name, data in sample.items():
        centered_by_level = data - tf.reduce_mean(data, axis=0)
        factor = tf.math.sqrt(tf.reduce_mean(centered_by_level ** 2))
        normalize_factors[name] = factor

    input_data = {name: avg_profiles[name] for name in model.input_names}

    with tf.GradientTape(persistent=True) as g:
        g.watch(input_data)
        outputs = model(input_data)

    all_jacobians = {}
    for out_name in model.output_names:
        per_input_jacobians = g.jacobian(outputs[out_name], input_data)

        normalized = {}
        for in_name, j in per_input_jacobians.items():
            # multiply d_output/d_input by std_input/std_output
            factor = normalize_factors[in_name] / normalize_factors[out_name]
            normalized[in_name] = (j[0, :, 0] * factor).numpy()

        all_jacobians[out_name] = normalized

    return all_jacobians


def plot_output_sensitivities(jacobians: Mapping[str, Mapping[str, tf.Tensor]]):

    """
    jacobians: mapping of each out variable to a sensitivity for each input
        e.g.,
        air_temperature_after_precpd:
            air_temperature_input: sensitivity matrix (nlev x nlev)
            specific_humidity_input: sensitivity matrix
        specific_humidity_after_precpd:
            ...
    """
    nrows = len(jacobians)
    per_input_example = next(iter(jacobians.values()))
    ncols = len(per_input_example)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        vertical_spacing=.15,
        row_titles=list(jacobians.keys()),
        column_titles=list(per_input_example.keys()),
        x_title="Input Variable",
        y_title="Output Variables",
    )

    for i, sensitivities in enumerate(jacobians.values(), 1):
        for j, sensitivity in enumerate(sensitivities.values(), 1):
            trace = go.Heatmap(z=sensitivity, coloraxis="coloraxis", zmin=-1, zmax=1)
            fig.append_trace(
                trace=trace, row=i, col=j,
            )

    fig.update_layout(
        title_text=f"Model Output Sensitivities ({config.model.architecture.name})",
        coloraxis={"colorscale": "RdBu_r", "cmax": 1, "cmin": -1},

    )
    fig.update_annotations(font_size=12)
    fig.write_html("/home/andrep/repos/fv3net/scratch/jacobian.html")

    return fig


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

    model = config.build(train_set)

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

    config.loss.prepare(output_samples=train_set)

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
                config.loss,
                epochs=config.epochs,
                validation_data=test_ds_batched,
                validation_freq=config.valid_freq,
                verbose=config.verbose,
                callbacks=callbacks,
            )

    logger.debug("Training complete")

    # with put_dir(config.out_url) as tmpdir:

    #     with open(os.path.join(tmpdir, "history.json"), "w") as f:
    #         json.dump(history.params, f)

    #     with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
    #         f.write(yaml.safe_dump(asdict(config)))

    #     local_model_path = save_model(model, tmpdir)

    #     if config.use_wandb:
    #         store_model_artifact(local_model_path, name=config._model.name)

    # Jacobians after model storing in case of "out of memory" errors
    jacobians = get_model_output_sensitivities(model, train_set)
    plot_output_sensitivities(jacobians)

    # with put_dir(config.out_url) as tmpdir:
    #     with open(os.path.join(tmpdir, "jacobians.npz"), "wb") as f:
    #         dumpable = {
    #             f"{out_name}/{in_name}": data
    #             for out_name, sensitivities in jacobians.items()
    #             for in_name, data in sensitivities.items()
    #         }
    #         np.savez(f, **dumpable)

    if config.use_wandb:
        sensitivity_plot = plot_output_sensitivities(jacobians)
        wandb.log({"output_sensitivity": wandb.Plotly(sensitivity_plot)})


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

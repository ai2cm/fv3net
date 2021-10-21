import argparse
import dacite
import dataclasses
import fsspec
import json
import os
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.wandb import WandBConfig, log_to_table, log_profile_plots, store_model_artifact
import wandb
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, List, Mapping, Optional, Sequence

from fv3fit.emulation.models import MicrophysicsConfig, ArchitectureConfig
from fv3fit.emulation.data import TransformConfig
from fv3fit.emulation.data import get_nc_files, nc_files_to_tf_dataset
from fv3fit.emulation.layers import MeanFeatureStdNormLayer≈
from fv3fit import set_random_seed

# TODO centralize this
from fv3fit.keras._models._filesystem import put_dir
from loaders.batches import shuffle


SCALE_VALUES = {
    "total_precipitation": 1000 / (900 / (3600 * 24)),  # mm / day
    "specific_humidity_output": 1000,  # g / kg
    "tendency_of_specific_humidity_due_to_microphysics": 1000
    * (3600 * 24),  # g / kg / day
    "cloud_water_mixing_ratio_output": 1000,  # g / kg
    "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": 1000
    * (3600 * 24),  # g / kg / day
    "air_temperature_output": 1,
    "tendency_of_air_temperature_due_to_microphysics": 3600 * 24,  # K / day,
}


# TODO: need to assert that outputs from data are same
#       order as all outputs from the model
@dataclasses.dataclass
class OptimizerConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.optimizers.Optimizer:
        cls = getattr(tf.keras.optimizers, self.name)
        return cls(**self.kwargs)


# TODO: update netcdf-> ds function
def _netcdf_url_to_dataset(url, transform, nfiles=None, do_shuffle=True):

    files = get_nc_files(url)
    if do_shuffle:
        files = shuffle(files)
    if nfiles is not None:
        files = files[:nfiles]

    return nc_files_to_tf_dataset(files, transform)


def get_out_samples(model_config: MicrophysicsConfig, samples, sample_names):

    # requires tendency_of_xxx_ for each residual to properly scale
    # the output denormalizer
    # enforce this in the configuration
    # right now the transform outputs matches model outputs

    direct_sample = []
    residual_sample = []

    for name in model_config.direct_out_variables:
        sample = samples[sample_names.index(name)]
        direct_sample.append(sample)

    for name in model_config.residual_out_variables:
        if name in model_config.tendency_outputs:
            tend_name = model_config.tendency_outputs[name]
            sample = samples[sample_names.index(tend_name)]
            residual_sample.append(sample)

    return direct_sample, residual_sample


def save_model(model: tf.keras.Model, destination: str):

    model.compiled_loss = None
    model.compiled_metrics = None
    model.optimizer = None
    model_path = os.path.join(destination, "model.tf")
    model.save(model_path, save_format="tf")

    return model_path


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    def __init__(self, sample_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalize = MeanFeatureStdNormLayer()
        self._normalize.fit(sample_data)

    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


def scale(names, values):

    scaled = []
    for name, value in zip(names, values):
        value *= SCALE_VALUES[name]
        scaled.append(value)

    return scaled


def score(target, prediction):

    bias_all = target - prediction
    se = bias_all ** 2

    mse = np.mean(se).astype(np.float)
    bias = np.mean(bias_all).astype(np.float)

    metrics = {
        "mse": mse,
        "bias": bias,
    }

    if target.ndim == 2 and target.shape[1] > 1:
        mse_prof = np.mean(se, axis=0)
        bias_prof = np.mean(bias_all, axis=0)
        rmse_prof = np.sqrt(mse_prof)

        profiles = {
            "mse_profile": mse_prof,
            "bias_profile": bias_prof,
            "rmse_profile": rmse_prof,
        }
    else:
        profiles = {}

    return metrics, profiles


def score_all(targets, predictions, names):

    all_scores = {}
    all_profiles = {}

    for target, pred, name in zip(targets, predictions, names):

        scores, profiles = score(target, pred)
        flat_score = {f"{k}/{name}": v for k, v in scores.items()}
        flat_profile = {f"{k}/{name}": v for k, v in profiles.items()}
        all_scores.update(flat_score)
        all_profiles.update(flat_profile)

    # assumes all profiles are same size
    profile = next(iter(all_profiles.values()))
    all_profiles["level"] = np.arange(len(profile))

    return all_scores, all_profiles


def load_config_yaml(path: str) -> Mapping[str, Any]:

    with fsspec.open(path, "r") as f:
        d = yaml.safe_load(f)

    return d


def to_flat_dict(d: dict):

    new_flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub_d = to_flat_dict(v)
            for sk, sv in sub_d.items():
                new_flat[".".join([k, sk])] = sv
        else:
            new_flat[k] = v

    return new_flat


def to_nested_dict(d: dict):

    new_config = {}

    for k, v in d.items():
        if "." in k:
            sub_keys = k.split(".")
            sub_d = new_config
            for sk in sub_keys[:-1]:
                sub_d = sub_d.setdefault(sk, {})
            sub_d[sub_keys[-1]] = v
        else:
            new_config[k] = v

    return new_config


def _add_items_to_parser_arguments(d: Mapping[str, Any], parser: argparse.ArgumentParser):

    for key, value in d.items():
        # TODO: should I do casting here, or let the dataclass do it?
        if not isinstance(value, str) and isinstance(value, Sequence):
            nargs = "*"
            default = tuple(value)
        else:
            nargs = None
            default = value
        parser.add_argument(f"--{key}", nargs=nargs, default=default)


@dataclasses.dataclass
class TrainConfig:
    train_url: str
    test_url: str
    out_url: str
    transform: TransformConfig
    model: MicrophysicsConfig
    wandb: Optional[WandBConfig] = None
    model_name: str = "microphysics-emulator"
    epochs: int = 1
    batch_size: int = 128
    nfiles: Optional[int] = None
    nfiles_valid: Optional[int] = None
    valid_freq: int = 5
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def asdict(self):
        return dataclasses.asdict(self)

    def as_flat_dict(self):
        return 

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainConfig":
        """Standard init from nested dictionary"""
        # casting necessary for 'from_args' which all come in as string
        # TODO: should this be just a json parsed??
        config = dacite.Config(strict=True, cast=[bool, str, int, float])
        return dacite.from_dict(cls, d, config=config)

    @classmethod
    def from_flat_dict(cls, d: Mapping[str, Any]) -> "TrainConfig":
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
    def from_args(cls, args: Sequence[Any] = None):
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
                "--model.architecure.name rnn"
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--config-path", type=str, default=None)

        args, unknown_args = parser.parse_known_args(args=args)

        if args.config_path == "default":
            config = get_default_config()
        elif args.config_path is not None:
            config = cls.from_yaml_path(args.config_path)
        else:
            raise ValueError(
                "No training configuration specified. Use '--config-path default' if just trying to run."
            )

        if unknown_args:
            updated = cls._get_updated_config_dict(unknown_args, config.as_flat_dict())
            config = cls.from_flat_dict(updated)

        return config

    @staticmethod
    def _get_updated_config_dict(args, flat_config_dict):

        config = dict(**flat_config_dict)
        parser = argparse.ArgumentParser()
        _add_items_to_parser_arguments(config, parser)
        updates = parser.parse_args(args)
        updates = vars(updates)

        config.update(updates)

        return config

    def __post_init__(self):

        if self.transform.input_variables != self.model.input_variables:
            raise ValueError(
                "Invalid training configuration state encountered. The"
                " data transform input variables ("
                f"{self.transform.input_variables}) are inconsistent with "
                f"the model input variables ({self.model.input_variables})."
            )
        elif self.transform.output_variables != self.model.output_variables:
            raise ValueError(
                "Invalid training configuration state encountered. The"
                " data transform output variables ("
                f"{self.transform.output_variables}) are inconsistent with "
                f"the model output variables ({self.model.output_variables})."
            )


def main(config: TrainConfig, seed: int = 0):
    set_random_seed(seed)

    if config.wandb is not None:
        config.wandb.init(config=config.asdict())

    train_ds = _netcdf_url_to_dataset(
        config.train_url, config.transform, nfiles=config.nfiles
    )
    test_ds = _netcdf_url_to_dataset(
        config.test_url, config.transform, nfiles=config.nfiles_valid
    )

    X_train, train_target = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    X_test, test_target = next(iter(test_ds.shuffle(160_000).batch(80_000)))
    direct_sample, resid_sample = get_out_samples(
        config.model, train_target, config.transform.output_variables
    )

    model = config.model.build(
        X_train, sample_direct_out=direct_sample, sample_residual_out=resid_sample
    )

    losses = {}
    metrics = {}
    weights = {}
    for out_varname, sample in zip(config.transform.output_variables, train_target):
        loss_func = NormalizedMSE(sample)
        if out_varname in config.loss_variables:
            losses[out_varname] = loss_func
            if out_varname in config.weights:
                weights[out_varname] = config.weights[out_varname]
            else:
                weights[out_varname] = 1.0
        elif out_varname in config.metric_variables:
            metrics[out_varname] = loss_func

    optimizer = config.optimizer.instance
    model.compile(
        loss=losses, metrics=metrics, optimizer=optimizer, loss_weights=weights
    )

    history = model.fit(
        train_ds.shuffle(100_000).batch(config.batch_size),
        epochs=config.epochs,
        validation_freq=config.valid_freq,
        validation_data=test_ds.batch(config.batch_size),
        verbose=2,
        callbacks=callbacks,
    )

    # scoring
    test_target = scale(config.transform.output_variables, test_target)
    train_target = scale(config.transform.output_variables, train_target)
    out_names = model.output_names
    test_pred = scale(out_names, model.predict(X_test))
    train_pred = scale(out_names, model.predict(X_train))

    train_scores, train_profiles = score_all(train_target, train_pred, out_names)
    test_scores, test_profiles = score_all(test_target, test_pred, out_names)

    if config.wandb is not None:
        log_to_table("score/train", train_scores, index=[config.wandb.job.name])
        log_to_table("score/test", test_scores, index=[config.wandb.job.name])
        log_to_table("profiles/train", train_profiles)
        log_to_table("profiles/test", test_profiles)

    with put_dir(config.out_url) as tmpdir:
        # TODO: need to convert ot np.float to serialize
        with open(os.path.join(tmpdir, "scores.json"), "w") as f:
            json.dump({"train": train_scores, "test": test_scores}, f)

        local_model_path = save_model(model, tmpdir)

        if config.wandb is not None:
            store_model_artifact(local_model_path, name=config.model.name)


def get_default_config():

    input_vars = [
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "pressure_thickness_of_atmospheric_layer",
    ]

    model_config = MicrophysicsConfig(
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
            air_temperature_output="tendency_of_air_temperature_due_to_microphysics",
            specific_humidity_output="tendency_of_specific_humidity_due_to_microphysics",
        ),
    )

    transform = TransformConfig(
        input_variables=input_vars, output_variables=model_config.output_variables,
    )

    config = TrainConfig(
        train_url="gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/training_netcdfs",  # noqa E501
        test_url="gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/validation_netcdfs",  # noqa E501
        out_url="gs://vcm-ml-scratch/andrep/test-train-emulation",
        model=model_config,
        transform=transform,
        nfiles=80,
        nfiles_valid=80,
        valid_freq=1,
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
        epochs=4,
        wandb=WandBConfig(job_type="training")
    )

    return config


if __name__ == "__main__":

    config = TrainConfig.from_args()
    main(config)

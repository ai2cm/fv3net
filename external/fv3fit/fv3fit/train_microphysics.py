import argparse
import copy
import dacite
import fsspec
import json
import os
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

from fv3fit import set_random_seed
from fv3fit._shared import put_dir
from fv3fit._shared.config import OptimizerConfig
from fv3fit.emulation.data import nc_dir_to_tf_dataset, TransformConfig
from fv3fit.emulation.data.config import SliceConfig
from fv3fit.emulation.keras import (
    CustomLoss,
    StandardLoss,
    save_model,
    score_model,
)
from fv3fit.emulation.models import MicrophysicsConfig, ArchitectureConfig
from fv3fit.wandb import (
    WandBConfig,
    log_to_table,
    log_profile_plots,
    store_model_artifact,
)


def _get_out_samples(model_config: MicrophysicsConfig, samples, sample_names):
    """
    Grab samples from a list separated into the direct output
    variables and residual output variables.  Used because the
    output normalization samples might need to be the field
    or the field tendencies
    """

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


def load_config_yaml(path: str) -> Dict[str, Any]:
    """
    Load yaml from local/remote location
    """

    with fsspec.open(path, "r") as f:
        d = yaml.safe_load(f)

    return d


def to_flat_dict(d: dict):
    """
    Converts any nested dictionaries to a flat version with
    the nested keys joined with a '.', e.g., {a: {b: 1}} ->
    {a.b: 1}
    """

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
    """
    Converts a flat dictionary with '.' joined keys back into
    a nested dictionary, e.g., {a.b: 1} -> {a: {b: 1}}
    """

    new_config: MutableMapping[str, Any] = {}

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


def _bool_from_str(value: str):
    affirmatives = ["y", "yes", "true", "t"]
    negatives = ["n", "no", "false", "f"]

    if value.lower() in affirmatives:
        return True
    elif value.lower() in negatives:
        return False
    else:
        raise ValueError(
            f"Unrecognized value encountered in boolean conversion: {value}"
        )


def _add_items_to_parser_arguments(
    d: Mapping[str, Any], parser: argparse.ArgumentParser
):
    """
    Take a dictionary and add all the keys as an ArgumentParser
    argument with the value as a default.  Does no casting so
    any non-defaults will likely be strings.  Instead relies on the
    dataclasses to do the validation and type casting.
    """

    for key, value in d.items():
        # TODO: should I do casting here, or let the dataclass do it?
        if isinstance(value, Mapping):
            raise ValueError(
                "Adding a mapping as an argument to the parse is not"
                " currently supported.  Make sure you are passing a"
                " 'flattened' dictionary to this function."
            )
        elif not isinstance(value, str) and isinstance(value, Sequence):
            kwargs = dict(nargs="*", default=copy.copy(value),)
        elif isinstance(value, bool):
            kwargs = dict(type=_bool_from_str, default=value)
        else:
            kwargs = dict(default=value)

        parser.add_argument(f"--{key}", **kwargs)


@dataclass
class TrainConfig:
    train_url: str
    test_url: str
    out_url: str
    transform: TransformConfig = field(default_factory=TransformConfig)
    model: MicrophysicsConfig = field(default_factory=MicrophysicsConfig)
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

    def asdict(self):
        return asdict(self)

    def as_flat_dict(self):
        return to_flat_dict(self.asdict())

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
                "--model.architecure.name rnn"
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

    callbacks = []
    if config.use_wandb:
        config.wandb.init(config=config.asdict())
        callbacks.append(config.wandb.get_callback())

    train_ds = nc_dir_to_tf_dataset(
        config.train_url, config.transform, nfiles=config.nfiles
    )
    test_ds = nc_dir_to_tf_dataset(
        config.test_url, config.transform, nfiles=config.nfiles_valid
    )

    X_train, train_target = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    X_test, test_target = next(iter(test_ds.shuffle(160_000).batch(80_000)))
    direct_sample, resid_sample = _get_out_samples(
        config.model, train_target, config.transform.output_variables
    )

    model = config.model.build(
        X_train, sample_direct_out=direct_sample, sample_residual_out=resid_sample
    )

    config.loss.prepare(output_names=model.output_names, output_samples=train_target)
    config.loss.compile(model)

    if config.shuffle_buffer_size is not None:
        train_ds = train_ds.shuffle(config.shuffle_buffer_size)

    history = model.fit(
        train_ds.batch(config.batch_size),
        epochs=config.epochs,
        validation_data=test_ds.batch(config.batch_size),
        validation_freq=config.valid_freq,
        verbose=config.verbose,
        callbacks=callbacks,
    )

    train_scores, train_profiles = score_model(model, X_train, train_target,)
    test_scores, test_profiles = score_model(model, X_test, test_target)

    if config.use_wandb:
        pred_sample = model.predict(X_test)
        log_profile_plots(test_target, pred_sample, model.output_names)

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
            f.write(yaml.safe_dump(config.asdict()))

        local_model_path = save_model(model, tmpdir)

        if config.use_wandb:
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
            air_temperature_output="tendency_of_air_temperature_due_to_microphysics",  # noqa E501
            specific_humidity_output="tendency_of_specific_humidity_due_to_microphysics",  # noqa E501
        ),
    )

    transform = TransformConfig(
        input_variables=input_vars, output_variables=model_config.output_variables,
    )

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

import dacite
import dataclasses
import json
import os
from fv3fit.emulation.data.config import convert_map_sequences_to_slices
import wandb
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, List, Mapping, MutableMapping, Optional

from fv3fit.emulation.microphysics import Config
from fv3fit.emulation.microphysics.models import ArchitectureParams
from fv3fit.emulation.data import TransformConfig
from fv3fit.emulation.data import get_nc_files, nc_files_to_tf_dataset
from fv3fit.emulation.layers import MeanFeatureStdNormLayer
from fv3fit.tensorboard import plot_to_image
from fv3fit import set_random_seed
# TODO centralize this
from fv3fit.keras._models._filesystem import put_dir
from loaders.batches import shuffle


SCALE_VALUES = {
    "total_precip": 1000/(3600*24),  # mm / day
    "specific_humidity_output": 1000,  # g / kg
    "tendency_of_specific_humidity_due_to_microphysics": 1000*(3600*24),  # g / kg / day
    "cloud_water_mixing_ratio_output": 1000,  # g / kg
    "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": 1000*(3600*24),  # g / kg / day
    "air_temperature_output": 1,
    "tendency_of_air_temperature_due_to_microphysics": 3600*24  # K / day
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


def get_out_samples(model_config: Config, samples, sample_names):

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
    se = bias_all**2

    mse = np.mean(se).astype(np.float)
    bias = np.mean(bias_all).astype(np.float)

    metrics = {
        "mse": mse,
        "bias": bias,
    }

    if target.ndim == 2:
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


def _log_profiles(targ, pred, name):

    for i in range(targ.shape[0]):
        levs = np.arange(targ.shape[1])
        fig = plt.figure()
        fig.set_size_inches(3, 5)
        fig.set_dpi(80)
        plt.plot(targ[i], levs, label="target")
        plt.plot(pred[i], levs, label="prediction")
        plt.title(f"Sample {i+1}: {name}")
        plt.xlabel(f"UNITS[name]")
        plt.ylabel("Level")
        wandb.log({f"sample_{i}_{name}": wandb.Image(plot_to_image(fig))})
        plt.close()


def log_sample_profiles(targets, predictions, names, nsamples=4):

    for targ, pred, name in zip(targets, predictions, names):

        targ = targ[:nsamples]
        pred = pred[:nsamples]

        if targ.ndim == 2:
            _log_profiles(targ, pred, name)


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


@dataclasses.dataclass
class TrainConfig:
    train_url: str
    test_url: str
    out_url: str
    transform: TransformConfig
    model: Config
    use_wandb: bool = True
    epochs: int = 1
    batch_size: int = 128
    nfiles: Optional[int] = None
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return dacite.from_dict(cls, d, dacite.Config(strict=True))

    @classmethod
    def from_yaml_path(cls, path: str) -> "TrainConfig":
        d = load_config_yaml(path)
        return cls.from_dict(d)

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


def _update_TrainConfig_model_selection(d: MutableMapping):
    """
    The model subselection field in the yaml is specified as
    a sequence of integers.  Need to convert those to slices
    and insert them into the dict if they are specified
    """

    d = dict(**d)
    if "model" in d:
        model_dict = d["model"]
        if "selection_map" in model_dict:
            d["model"]["selection_map"] = \
                 convert_map_sequences_to_slices(model_dict["selection_map"])

    return d


def load_config_yaml(path: str) -> Mapping[str, Any]:

    with open(path, "r") as f:
        d = yaml.safe_load(f)

    d = _update_TrainConfig_model_selection(d)

    return d


def main(config: TrainConfig):
    set_random_seed(0)

    callbacks = []
    if config.use_wandb:
        job = wandb.init(
            entity="ai2cm", project="microphysics-emulation-test", job_type="training",
        )
        # saves best model by validation every epoch
        callbacks.append(wandb.keras.WandbCallback())
    else:
        job = None

    train_ds = _netcdf_url_to_dataset(config.train_url, config.transform, nfiles=config.nfiles)
    test_ds = _netcdf_url_to_dataset(config.test_url, config.transform, nfiles=config.nfiles)

    sample_in, sample_out = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    direct_sample, resid_sample = get_out_samples(
        config.model, sample_out, config.transform.output_variables
    )
    model = config.model.build(
        sample_in, sample_direct_out=direct_sample, sample_residual_out=resid_sample
    )

    losses = {}
    metrics = {}
    for out_varname, sample in zip(config.transform.output_variables, sample_out):
        loss_func = NormalizedMSE(sample)
        if out_varname in config.loss_variables:
            losses[out_varname] = loss_func
        elif out_varname in config.metric_variables:
            metrics[out_varname] = loss_func

    optimizer = config.optimizer.instance
    model.compile(loss=losses, metrics=metrics, optimizer=optimizer)

    history = model.fit(
        train_ds.shuffle(100_000).batch(config.batch_size),
        epochs=config.epochs,
        validation_freq=1,
        validation_data=test_ds.batch(config.batch_size),
        verbose=2,
        callbacks=callbacks,
    )

    # scoring
    X_test, target = next(iter(test_ds.shuffle(160_000).batch(80_000)))
    target = scale(config.transform.output_variables, target)
    out_names = model.output_names
    test_pred = scale(out_names, model.predict(X_test))
    train_pred = scale(out_names, model.predict(sample_in))

    train_scores, train_profiles = score_all(target, train_pred, out_names)
    test_scores, test_profiles = score_all(target, test_pred, out_names)

    if config.use_wandb:
        # TODO: log scores func
        train_df = pd.DataFrame(train_scores, index=[job.name])
        train_table = wandb.Table(dataframe=train_df)
        test_df = pd.DataFrame(test_scores, index=[job.name])
        test_table = wandb.Table(dataframe=test_df)
        test_prof_table = wandb.Table(dataframe=pd.DataFrame(test_profiles))
        train_prof_table = wandb.Table(dataframe=pd.DataFrame(train_profiles))
        job.log({
            "score/train": train_table,
            "score/test": test_table,
            "profiles/test": test_prof_table,
            "profiles/train": train_prof_table,
        })

        log_sample_profiles(target, test_pred, out_names)

    if config.out_url:
        with put_dir(config.out_url) as tmpdir:
            # TODO: need to convert ot np.float to serialize
            with open(os.path.join(tmpdir, "scores.json"), "w") as f:
                json.dump({"train": train_scores, "test": test_scores}, f)

            model.compiled_loss = None
            model.compiled_metrics = None
            model.optimizer = None
            model_dir = os.path.join(tmpdir, "model.tf")
            model.save(model_dir, save_format="tf")

            if config.use_wandb:
                name = config.model.architecture.name
                model = wandb.Artifact(
                    f"microphysics-emulator-{name}",
                    type="model"
                )
                model.add_dir(model_dir)
                wandb.log_artifact(model)


def get_default_config():

    input_vars = [
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "pressure_thickness_of_atmospheric_layer",
    ]

    model_config = Config(
        input_variables=input_vars,
        direct_out_variables=[
            "cloud_water_mixing_ratio_output",
            "total_precipitation",
        ],
        residual_out_variables=dict(
            air_temperature_output="air_temperature_input",
            specific_humidity_output="specific_humidity_input",
        ),
        architecture=ArchitectureParams("linear"),
        selection_map=dict(
            air_temperature_input=slice(None, -10),
            specific_humidity_input=slice(None, -10),
            cloud_water_mixing_ratio_input=slice(None, -10),
            pressure_thickness_of_atmospheric_layer=slice(None, -10),
        ),
        tendency_outputs=dict(
            air_temperature_output="tendency_of_air_temperature_due_to_microphysics",
            specific_humidity_output="tendency_of_specific_humidity_due_to_microphysics",
        )
    )

    transform = TransformConfig(
        input_variables=input_vars,
        output_variables=model_config.output_variables,
    )

    config = TrainConfig(
        train_url="/mnt/disks/scratch/microphysics_emu_data/training_netcdfs/",
        test_url="/mnt/disks/scratch/microphysics_emu_data/validation_netcdfs/",
        out_url="/mnt/disks/scratch/test_train_out/",
        model=model_config,
        transform=transform,
        optimizer=OptimizerConfig(name="Adam", kwargs=dict(learning_rate=1e-4)),
        loss_variables=[
            "air_temperature_output",
            "specific_humidity_output",
            "cloud_water_mixing_ratio_output",
            "total_precipitation"
        ],
        metric_variables=[
            "tendency_of_air_temperature_due_to_microphysics",
            "tendency_of_specific_humidity_due_to_microphysics",
            "tendency_of_cloud_water_mixing_ratio_due_to_microphysics",
        ],
        epochs=4,
        nfiles=10,
        use_wandb=True,
    )

    return config


if __name__ == "__main__":
    main(get_default_config())

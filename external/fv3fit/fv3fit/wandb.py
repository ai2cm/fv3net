import dataclasses
from typing import Any, Mapping, Optional
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tensorboard import plot_to_image


# TODO: currently coupled to the scaling

UNITS = {
    "total_precipitation": "mm / day",
    "specific_humidity_output": "g/kg",
    "tendency_of_specific_humidity_due_to_microphysics": "g/kg/day",
    "cloud_water_mixing_ratio_output": "g/kg",
    "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": "g/kg/day",
    "air_temperature_output": "K",
    "tendency_of_air_temperature_due_to_microphysics": "K/day",
}


@dataclasses.dataclass
class WandBConfig:

    entity: str = "ai2cm"
    job_type: str = "test"
    wandb_project: str = "scratch-project"
    job = dataclasses.field(init=False)

    @staticmethod
    def get_callback():
        return wandb.keras.WandbCallback(save_weights_only=False)

    def init(self, config: Optional[Mapping[str, Any]] = None):
        self.job = wandb.init(
            entity=self.entity,
            project=self.wandb_project,
            job_type=self,
            config=config,
        )


def log_to_table(log_key, data, index=None):

    df = pd.DataFrame(data, index=index)
    table = wandb.Table(dataframe=df)

    wandb.log({log_key: table})


def _plot_profiles(target, prediction, name):

    nsamples = target.shape[0]

    for i in range(nsamples):
        levs = np.arange(target.shape[1])
        fig = plt.figure()
        fig.set_size_inches(3, 5)
        fig.set_dpi(80)
        plt.plot(target[i], levs, label="target")
        plt.plot(prediction[i], levs, label="prediction")
        plt.title(f"Sample {i+1}: {name}")
        plt.xlabel(f"{UNITS[name]}")
        plt.ylabel("Level")
        wandb.log({f"{name}_sample_{i}": wandb.Image(plot_to_image(fig))})
        plt.close()


def log_profile_plots(targets, predictions, names):

    if len(names) == 1:
        _plot_profiles(targets, predictions, names[0])
    else:
        for t, p, name in zip(targets, predictions, names):
            _plot_profiles(t, p, name)


def store_model_artifact(dir: str, name: str):

    model_artifact = wandb.Artifact(name, type="model")
    model_artifact.add_dir(dir)
    wandb.log_artifact(model_artifact)

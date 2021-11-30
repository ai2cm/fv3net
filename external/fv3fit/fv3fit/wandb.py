import dataclasses
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional

from .tensorboard import plot_to_image


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
    """
    Basic configuration for a  Weights & Biases job

    Args:
        entity: who's logging the data
        job_type: name to separate types of work under a project
        wandb_project: project name to store runs under
    """

    entity: str = "ai2cm"
    job_type: str = "test"
    wandb_project: str = "scratch-project"

    @staticmethod
    def get_callback():
        """Grab a callback for logging during keras training"""
        return wandb.keras.WandbCallback(save_weights_only=False)

    def init(self, config: Optional[Dict[str, Any]] = None):
        """Start logging specified by this config"""
        self.job = wandb.init(
            entity=self.entity,
            project=self.wandb_project,
            job_type=self.job_type,
            config=config,
        )


def log_to_table(log_key: str, data: Dict[str, Any], index: Optional[List[Any]] = None):
    """
    Log data to wandb table using a pandas dataframe
    
    Args:
        log_key: wandb key for table artifact
        data: mapping of data, each key becomes a dataframe
            column
        index: Provide an index for the column values, useful when
            the data are single values, e.g., the scoring metrics
    """

    df = pd.DataFrame(data, index=index)
    table = wandb.Table(dataframe=df)

    wandb.log({log_key: table})


def _plot_profiles(target, prediction, name):
    """Plot vertical profile comparisons"""

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


def log_profile_plots(
    targets: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    nsamples: int = 4,
):
    """
    Handle profile plots for single- or multi-output models.
    """
    for name in set(targets) & set(predictions):
        t = targets[name]
        p = predictions[name]
        _plot_profiles(t[:nsamples], p[:nsamples], name)


def store_model_artifact(path: str, name: str):
    """
    Store a tf model directory as a WandB artifact
    
    Args:
        path: Path to tensorflow saved model (e.g., /path/to/model.tf/)
        name: name for the WandB artifact.  If it already exists a new
            version is stored
    """

    model_artifact = wandb.Artifact(name, type="model")
    model_artifact.add_dir(path)
    wandb.log_artifact(model_artifact)

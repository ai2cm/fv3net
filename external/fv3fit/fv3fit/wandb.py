from collections import defaultdict
import dataclasses
import logging
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Mapping, Optional

from .tensorboard import plot_to_image
from ._shared.jacobian import OutputSensitivity


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
        # disable model saving since wandb doesn't support saving models in
        # SavedModel format and our models only support that format
        # see https://github.com/wandb/client/issues/2987
        return wandb.keras.WandbCallback(save_model=False)

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
    logging.getLogger("log_to_table").debug(f"logging {log_key} to weights and biases")

    df = pd.DataFrame(data, index=index)
    table = wandb.Table(dataframe=df)

    wandb.log({log_key: table})


def _plot_profiles(target, prediction, name):
    """Plot vertical profile comparisons"""

    units = defaultdict(lambda: "unknown")
    units.update(
        {
            "total_precipitation": "mm / day",
            "specific_humidity_output": "g/kg",
            "tendency_of_specific_humidity_due_to_microphysics": "g/kg/day",
            "cloud_water_mixing_ratio_output": "g/kg",
            "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": "g/kg/day",
            "air_temperature_output": "K",
            "tendency_of_air_temperature_due_to_microphysics": "K/day",
        }
    )

    nsamples = target.shape[0]

    for i in range(nsamples):
        levs = np.arange(target.shape[1])
        fig = plt.figure()
        fig.set_size_inches(3, 5)
        fig.set_dpi(80)
        plt.plot(target[i], levs, label="target")
        plt.plot(prediction[i], levs, label="prediction")
        plt.title(f"Sample {i+1}: {name}")
        plt.xlabel(f"{units[name]}")
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
    logging.getLogger("log_profile_plots").debug(
        f"logging {list(targets)} to weights and biases"
    )
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


def _plot_single_output_sensitivities(
    name: str, jacobians: OutputSensitivity
) -> go.Figure:

    ncols = len(jacobians)
    fig = make_subplots(
        rows=1, cols=ncols, shared_yaxes=True, column_titles=list(jacobians.keys())
    )

    for j, sensitivity in enumerate(jacobians.values(), 1):
        trace = go.Heatmap(z=sensitivity, coloraxis="coloraxis", zmin=-1, zmax=1)
        fig.append_trace(
            trace=trace, row=1, col=j,
        )
        fig.update_xaxes(title_text="Input Level", row=1, col=j)
    fig.update_yaxes(title_text="Output Level", row=1, col=1)

    fig.update_layout(
        title_text=f"Standardized {name} input sensitivities",
        coloraxis={"colorscale": "RdBu_r", "cmax": 1, "cmin": -1},
        height=400,
    )

    return fig


def plot_all_output_sensitivities(jacobians: Mapping[str, OutputSensitivity]):

    """
    Create a plotly heatmap for each input sensitivity matrix for each model
    output.

    jacobians: mapping of each out variable to a sensitivity for each input
        e.g.,
        air_temperature_after_precpd:
            air_temperature_input: sensitivity matrix (nlev x nlev)
            specific_humidity_input: sensitivity matrix
        specific_humidity_after_precpd:
            ...
    """

    all_plots = {
        out_name: _plot_single_output_sensitivities(out_name, out_sensitivities)
        for out_name, out_sensitivities in jacobians.items()
    }

    for out_name, fig in all_plots.items():
        wandb.log({f"jacobian/{out_name}": wandb.Plotly(fig)})

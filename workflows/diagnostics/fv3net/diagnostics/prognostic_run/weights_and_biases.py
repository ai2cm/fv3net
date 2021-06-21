import fv3viz
from matplotlib import pyplot as plt
import wandb
import fsspec

from fv3net.diagnostics.prognostic_run import computed_diagnostics
from fv3net.artifacts.query import get_artifacts, Step
import vcm
import numpy as np


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "wandb",
        description="Hint: use environmental variables to configure "
        "the target W and B project and entity: "
        "https://docs.wandb.ai/guides/track/advanced/environment-variables",
    )
    parser.set_defaults(func=main)


def select_by_dims(ds, dims):
    var = [v for v in ds if set(ds[v].dims) == set(dims)]
    return ds[var]


def upload(step: Step):
    """Upload metrics information to Weights and Biases

    Raises:
        FileNotFoundError
    """

    # run = api.runs(
    #     "ai2cm/prognostic-run-testing", filters={"config.path": str(step.path)}
    # )

    # 1. Start a W&B run
    # wandb.init(resume=True, id=run[0].id if run else None, allow_val_change=True)
    wandb.init(
        name=step.path.as_posix(), entity="ai2cm", project="prognostic-run-testing"
    )

    # # 2. Save model inputs and hyperparameters
    fs = fsspec.filesystem(step.fs.protocol[1])
    folder = computed_diagnostics.DiagnosticFolder(fs, step.path)
    config = wandb.config
    config.path = folder.path
    config.project = step.project
    config.date_created = step.date_created
    config.tag = step.tag

    # # 3. Log metrics over time to visualize performance
    metrics = folder.metrics
    metrics_no_units = {key: metrics[key]["value"] for key in metrics}
    wandb.log(metrics_no_units)

    # upload time series
    diagnostics = folder.diagnostics
    diagnostics["time"] = np.vectorize(lambda x: x.isoformat())(
        np.vectorize(vcm.cast_to_datetime)(diagnostics.time)
    )
    df = select_by_dims(diagnostics, ["time"]).to_dataframe().reset_index()
    wandb.log({"time_dependent": wandb.Table(data=df)})

    # plot maps
    maps_2d = select_by_dims(diagnostics, ["x", "y", "tile"])
    variables = list(maps_2d)
    maps_2d["lonb"] = diagnostics.lonb
    maps_2d["latb"] = diagnostics.latb
    for v in variables:
        print(v)
        fig = fv3viz.plot_cube(maps_2d, v)[0]
        fig.suptitle(v)
        wandb.log({v: wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()


def main(_):
    artifacts = get_artifacts(
        "gs://vcm-ml-experiments", projects=["default", "spencerc"]
    )
    for artifact in artifacts:
        if artifact.step == "fv3gfs_run_diagnostics":
            try:
                upload(artifact)
            except FileNotFoundError:
                pass

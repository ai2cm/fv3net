from pathlib import Path
import tempfile
from typing import Optional
from enum import Enum

import wandb
import xarray
import subprocess
import fv3config
import os

from .prognostic import compute_metrics, open_run
from .config import get_config

import typer

app = typer.Typer()


def log_vertical_metrics(key, metrics: xarray.Dataset):
    df = metrics.to_dataframe().reset_index()
    df["time"] = df.time.apply(lambda x: x.isoformat())
    wandb.log({key: wandb.Table(dataframe=df)})


def log_summary_metrics(label: str, mean: xarray.Dataset):
    for key in mean:
        wandb.summary[label + "/" + str(key)] = float(mean[key])


def run(config, path):
    with tempfile.NamedTemporaryFile("w") as user_config:
        fv3config.dump(config, user_config)
        subprocess.check_call(
            ["runfv3", "create", path, user_config.name, "sklearn_runfile.py"]
        )
    subprocess.check_call(["runfv3", "append", path])


def evaluate(path):
    ds = open_run(path)
    metrics = compute_metrics(ds)
    log_vertical_metrics("vertical_metrics", metrics)
    log_summary_metrics("mean", metrics.mean())
    wandb.finish()


class Mask(str, Enum):
    default = "default"
    model_2021_09_16 = "2021_09_16"


def short(
    artifact_id: str,
    online: bool = True,
    mask: Mask = Mask.default,
    config_updates: Optional[Path] = Path("config/3-hour.yaml"),
):
    job = wandb.init(entity="ai2cm", project="emulator-noah", job_type="prognostic-run")
    if job is None:
        raise RuntimeError("Weights and biases not initialized properly.")

    path = Path("/data/prognostic-runs") / str(job.id)
    artifact = wandb.use_artifact(artifact_id)
    artifact_path = os.path.abspath(artifact.download())

    config = get_config(suite=None, diagnostics=config_updates)
    config["online_emulator"] = {
        "emulator": artifact_path,
        "online": online,
        "train": False,
        "mask_kind": mask.value,
    }
    wandb.config.update(config)

    run(config, path)
    evaluate(path)


if __name__ == "__main__":
    typer.run(short)

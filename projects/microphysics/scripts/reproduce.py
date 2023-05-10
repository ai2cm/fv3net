#!/usr/bin/env python3
from typing import Any
import typer
import wandb
import end_to_end

app = typer.Typer()

PROJECT = "ai2cm/microphysics-emulation"


def wandb2job(run: Any) -> end_to_end.ToYaml:
    if run.job_type == "prognostic_run":
        return end_to_end.PrognosticJob(
            name=run.name,
            config=run.config["config"],
            image_tag=run.config["env"]["COMMIT_SHA"],
            project="microphysics-emulation",
            bucket="vcm-ml-experiments",
        )
    if run.job_type == "train":
        config = dict(run.config)
        env = config.pop("env")
        sha = env["COMMIT_SHA"]
        return end_to_end.TrainingJob(
            name=run.name,
            config=config,
            fv3fit_image_tag=sha,
            project="microphysics-emulation",
            bucket="vcm-ml-experiments",
        )
    else:
        raise NotImplementedError(f"{run.job_type} not implemented.")


@app.command()
def reproduce(run_id: str):
    """Show the yaml required to reproduce a given wadnb run"""
    api = wandb.Api()
    run = api.run(run_id)
    job = wandb2job(run)
    print(job.to_yaml())


if __name__ == "__main__":
    app()

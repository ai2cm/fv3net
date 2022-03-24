import sys
import wandb
import dataclasses

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402


@dataclasses.dataclass
class TrainedModel:
    path: str
    name: str


def _get_job(model: TrainedModel):
    config = load_yaml("../configs/default.yaml")
    config["zhao_carr_emulation"] = {"model": {"path": model.path}}
    return PrognosticJob(
        name=f"{model.name}-30d-v1",
        image_tag="d848e586db85108eb142863e600741621307502b",
        config=config,
    )


EXPERIMENT_TAG = "experiment/more-normalization"
WANDB = "ai2cm/microphysics-emulation"


def yield_model_jobs():
    api = wandb.Api()
    for job in api.runs(WANDB, filters={"tags": EXPERIMENT_TAG, "state": "finished"}):
        if job.job_type == "train" and ("bug" not in job.tags):
            yield TrainedModel(job.config["out_url"], job.name)


jobs = [_get_job(job) for job in yield_model_jobs()]
submit_jobs(jobs, "more-normalization-30d")

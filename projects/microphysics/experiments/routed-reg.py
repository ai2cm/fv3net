import sys

sys.path.insert(0, "../argo")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


def _get_job():
    config = load_yaml(f"../train/gscond-only-routed.yaml")
    config["epochs"] = 25
    return TrainingJob(name=name, fv3fit_image_tag=revision, config=config,)


name = "gscond-routed-reg-v3"
revision = "8cdac54391290cf256af089be941917165e4705f"
submit_jobs([_get_job()], f"routed-reg")

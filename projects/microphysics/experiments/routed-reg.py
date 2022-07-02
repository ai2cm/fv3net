import sys

sys.path.insert(0, "../argo")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


def _get_job():
    config = load_yaml(f"../train/gscond-only-routed.yaml")
    config["epochs"] = 25
    return TrainingJob(name=name, fv3fit_image_tag=revision, config=config,)


name = "gscond-routed-reg-v4"
revision = "5003e52966b17f64be2da40f0dde1d70c95088a7"
submit_jobs([_get_job()], f"routed-reg")

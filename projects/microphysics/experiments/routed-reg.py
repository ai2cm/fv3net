import sys

sys.path.insert(0, "../argo")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


def _get_job():
    config = load_yaml(f"../train/gscond-only-routed.yaml")
    config["epochs"] = 25
    return TrainingJob(
        name=f"gscond-routed-reg-v1", fv3fit_image_tag=revision, config=config,
    )


revision = "d51594567be2297b141d89458dded5942cf53440"
submit_jobs([_get_job()], f"routed-reg")

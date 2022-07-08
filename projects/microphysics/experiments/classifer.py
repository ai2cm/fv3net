import sys

sys.path.insert(0, "../argo")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


def _get_job(revision):
    config = load_yaml(f"../train/gscond-precpd-classifier.yaml")
    # Training loss problems are visible after 5 epochs
    config["epochs"] = 50
    return TrainingJob(
        name=f"gscond-precpd-classifier-v1", fv3fit_image_tag=revision, config=config,
    )


image_tag = "fa9d94983c8b89fac75654260ae6b9fc114f4dff"
submit_jobs([_get_job(image_tag)], f"classifier")

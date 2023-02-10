import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    TrainingJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


CONFIG_PATHS = {
    "gscond": "../train/gscond-only.yaml",
    "gscond-classify": "../train/gscond-classifier.yaml",
    "precpd": "../train/rnn-precpd-diff-only.yaml",
}
NAME = "zc-train-{}-seed{}-v2"
IMAGE_TAG = "353e842a653200175a219cd6ad3c1a45ffbc2e13"


def get_jobs(seed: int):

    for key, cfg_path in CONFIG_PATHS.items():
        cfg = load_yaml(cfg_path)["config"]
        cfg["seed"] = seed

        name = NAME.format(key, seed)
        yield TrainingJob(config=cfg, name=name, fv3fit_image_tag=IMAGE_TAG)


jobs = []
for i in range(1, 6):
    jobs.extend(get_jobs(i))

submit_jobs(jobs, experiment_name="seed-sensitivity")

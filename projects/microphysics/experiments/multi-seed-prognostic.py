import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


BASE = "gs://vcm-ml-experiments/microphysics-emulation/2022-12-16/zc-train-{model}-seed{i}-v2/model.tf"  # noqa
NAME = "zc-emu-normfix-seed{}-prognostic-30d-v1"

# Use normalization fix gscond models
NORM_FIX = True
SEED5 = "gs://vcm-ml-experiments/microphysics-emulation/2023-02-06/gscond-only-qvout-norm-fix-v1/model.tf"  # noqa
SEED0_4 = "gs://vcm-ml-experiments/microphysics-emulation/2023-02-10/zc-train-gscond-normfix-seed{i}-v1/model.tf"  # noqa


def get_job(seed: int):

    cfg = load_yaml("../configs/gscond-and-precpd.yaml")["config"]
    cfg["duration"] = "30d"
    zc_config = cfg["zhao_carr_emulation"]

    if NORM_FIX:
        gscond_path = SEED0_4.format(i=seed)
    else:
        gscond_path = BASE.format(model="gscond", i=seed)

    if seed == 0:
        classify_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/gscond-only-classifier-rh-in-v1/model.tf"  # noqa
        precpd_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/precpd-diff-only-press-rh-in-v1/model.tf"  # noqa
    else:
        classify_path = BASE.format(model="gscond-classify", i=seed)
        precpd_path = BASE.format(model="precpd", i=seed)

    zc_config["gscond"]["path"] = gscond_path
    zc_config["gscond"]["classifier_path"] = classify_path
    zc_config["model"]["path"] = precpd_path

    return PrognosticJob(
        config=cfg,
        image_tag="95218aec9ed1e6cbda0db1e1f69dc34e0d839c52",
        name=NAME.format(seed),
    )


if NORM_FIX:
    use_seeds = range(5)
else:
    use_seeds = range(1, 6)

jobs = [get_job(i) for i in use_seeds]

submit_jobs(jobs, experiment_name="seed-sensitivity")

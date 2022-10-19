import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402

model_tags = [
    ["rh-in"] * 2,
    ("press-in-scale-adjust", "in-scale-adjust"),
    ["rh-in-and-pscale-adj"] * 2,
]

CLASSIFIER = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/gscond-only-classifier-{}-v1/model.tf"  # noqa
GSCOND = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/gscond-only-dense-{}-v1/model.tf"  # noqa
PRECPD = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/precpd-diff-only-press-{}-v1/model.tf"  # noqa
NAME = "zc-emu-{}-v1"


def get_job(model_tags):

    gscond, precpd = model_tags

    config = load_yaml("../configs/combined-base.yaml")

    config["duration"] = "6d"
    config["zhao_carr_emulation"]["gscond"]["path"] = GSCOND.format(gscond)
    config["zhao_carr_emulation"]["gscond"]["classifier_path"] = CLASSIFIER.format(
        gscond
    )
    config["zhao_carr_emulation"]["model"]["path"] = PRECPD.format(precpd)

    return PrognosticJob(
        name=NAME.format(gscond),
        image_tag="latest",
        config=config,
        fv3net_image_tag="9d2404bb4f04a3c7645692f3b05670b8ed1c2c73",
    )


jobs = [get_job(t) for t in model_tags]
submit_jobs(jobs, f"rh-trials-prognostic-oct2022")

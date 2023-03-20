import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


dense_model_paths = {
    "gscond": "gs://vcm-ml-experiments/microphysics-emulation/2023-03-17/gscond-only-qvout-norm-fix-dense-v1/model.tf",  # noqa
    "precpd": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/precpd-diff-only-dense-no-pr-v2/model.tf",  # noqa
}
model_paths = {
    "gscond": "gs://vcm-ml-experiments/microphysics-emulation/2023-02-10/zc-train-gscond-normfix-seed3-v1/model.tf",  # noqa
    "precpd": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-16/zc-train-precpd-seed5-v2/model.tf",  # noqa
}

dense_precpd_paths = {
    "gscond": "gs://vcm-ml-experiments/microphysics-emulation/2023-02-10/zc-train-gscond-normfix-seed3-v1/model.tf",  # noqa,
    "precpd": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/precpd-diff-only-dense-no-pr-v2/model.tf",  # noqa
}


def get_job(model_paths, name):
    config = load_yaml("../configs/gscond-and-precpd.yaml")["config"]

    config["duration"] = "30d"
    config["zhao_carr_emulation"]["gscond"]["path"] = model_paths["gscond"]
    config["zhao_carr_emulation"]["model"]["path"] = model_paths["precpd"]

    return PrognosticJob(
        name=name, image_tag="95218aec9ed1e6cbda0db1e1f69dc34e0d839c52", config=config,
    )


jobs = [
    get_job(dense_model_paths, "zc-emu-using-dense-v3"),
    get_job(model_paths, "zc-emu-not-dense-v3"),
    get_job(dense_precpd_paths, "zc-emu-gsc-prior-precpd-simple-v3"),
]
submit_jobs(jobs, "compare-dense-mar2023")

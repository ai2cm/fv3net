import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


dense_model_paths = {
    "gscond": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/gscond-only-dense-qvout-v2/model.tf",  # noqa
    "precpd": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/precpd-diff-only-dense-no-pr-v2/model.tf",  # noqa
}
model_paths = {
    "gscond": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/gscond-only-qvout-v2/model.tf",  # noqa
    "precpd": "gs://vcm-ml-experiments/microphysics-emulation/2022-12-13/precpd-diff-only-no-pr-v2/model.tf",  # noqa
}


def get_job(model_paths, name):
    config = load_yaml("../configs/gscond-and-precpd.yaml")["config"]

    config["duration"] = "30d"
    config["zhao_carr_emulation"]["gscond"]["path"] = model_paths["gscond"]
    config["zhao_carr_emulation"]["model"]["path"] = model_paths["precpd"]

    return PrognosticJob(
        name=name, image_tag="99b74dcc46b233c05942b32098c1207b0e6dd323", config=config,
    )


jobs = [
    get_job(dense_model_paths, "zc-emu-using-dense-v2"),
    get_job(model_paths, "zc-emu-not-dense-v2"),
]
submit_jobs(jobs, "compare-offline-dense-dec2022")

import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


BASE = "gs://vcm-ml-experiments/microphysics-emulation/2022-12-16/zc-train-{model}-seed{i}-v2/model.tf"  # noqa
NAME = "zc-emu-seed{}-prognostic-30d-v2"


def get_job(seed: int):

    cfg = load_yaml("../configs/gscond-and-precpd.yaml")["config"]
    cfg["duration"] = "30d"
    zc_config = cfg["zhao_carr_emulation"]

    zc_config["gscond"]["path"] = BASE.format(model="gscond", i=seed)
    zc_config["gscond"]["classifier_path"] = BASE.format(
        model="gscond-classify", i=seed
    )
    zc_config["model"]["path"] = BASE.format(model="precpd", i=seed)

    return PrognosticJob(config=cfg, image_tag="latest", name=NAME.format(seed))


jobs = [get_job(i) for i in range(1, 6)]

submit_jobs(jobs, experiment_name="seed-sensitivity")

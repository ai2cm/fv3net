import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
    set_prognostic_emulation_model,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-dense-local-nfiles3960-41b1c1-v1/model.tf"  # noqa
NAME = "fixtenddiags-model273r8qbx-v2"
config = load_yaml("../configs/default.yaml")

config = set_prognostic_emulation_model(
    config, MODEL, gscond_only=True, gscond_conservative=True
)
config["duration"] = "2d"

job = PrognosticJob(
    name=NAME, image_tag="324cd5c40b4d0f5b756828091904e34125f51b4e", config=config,
)

jobs = [job]
submit_jobs(jobs, f"gscond-only-emu-may2022")

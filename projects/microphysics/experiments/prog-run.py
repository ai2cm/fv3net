import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1/model.tf"  # noqa
NAME = "gscond-only-dense-nfile1980-upper5-2d-v1"

config = load_yaml("../configs/default.yaml")

config["duration"] = "2d"

zc_config = {"gscond": {"path": MODEL, "gscond_cloud_conservative": True}}
config["zhao_carr_emulation"] = zc_config
config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = True

job = PrognosticJob(name=NAME, image_tag="latest", config=config,)

jobs = [job]
submit_jobs(jobs, f"gscond-only-emu-may2022")

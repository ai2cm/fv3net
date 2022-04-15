import sys

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-all-loss-rnn-7ef273/model.tf"  # noqa: E501
config = load_yaml("../configs/default.yaml")

# set gscond only model
config["zhao_carr_emulation"] = {"gscond": {"path": MODEL}}
config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = True

job = PrognosticJob(
    name=f"squash-control-gscond-only", image_tag="latest", config=config
)

jobs = [job]
submit_jobs(jobs, f"squash")

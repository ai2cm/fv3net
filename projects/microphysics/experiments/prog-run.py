import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
    set_prognostic_emulation_model,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1/model.tf"  # noqa: E501
config = load_yaml("../configs/default.yaml")

# Stop emulator application on top 5 model levels
upper_5_levels = dict(start=74, stop=None)
mask_levels = {
    "specific_humdity_after_gscond": upper_5_levels,
    "air_temperature_after_gscond": upper_5_levels,
}

model_kwargs = dict(gscond_cloud_conservative=True, mask_emulator_levels=mask_levels,)
config = set_prognostic_emulation_model(
    config, MODEL, gscond_only=True, emu_model_kwargs=model_kwargs,
)

job = PrognosticJob(
    name=f"gscond-only-tscale-dense-local-41b1c1-v1-30d",
    image_tag="41b1c1a5443ebb5903ed61bd6923cc628b4981a2",
    config=config,
)

jobs = [job]
submit_jobs(jobs, f"gscond-only-emu-may2022")

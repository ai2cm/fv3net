import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
    set_prognostic_emulation_model,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1/model.tf"  # noqa
CLASSIFIER = "gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1/model.tf"  # noqa

mask_levels = {
    "cloud_water_mixing_ratio_after_gscond": dict(start=74, stop=None),
    "specific_humidity_after_gscond": dict(start=74, stop=None),
    "air_temperature_after_gscond": dict(start=74, stop=None),
}


def _get_job():
    config = load_yaml("../configs/default.yaml")

    config = set_prognostic_emulation_model(
        config, MODEL, gscond_only=True, gscond_conservative=True
    )
    gscond_config = config["zhao_carr_emulation"]["gscond"]
    gscond_config["classifier_path"] = CLASSIFIER
    gscond_config["mask_emulator_levels"] = mask_levels
    gscond_config["mask_gscond_zero_cloud_classifier"] = True
    gscond_config["mask_gscond_zero_tend_classifier"] = True
    gscond_config["enforce_conservative"] = True

    return PrognosticJob(
        name=f"gscond-only-classifier-zcloud-ztend-online-10d-v1",
        image_tag="ea5443c7b008f6435cec24c32132be35a1613204",
        config=config,
    )


submit_jobs([_get_job()], f"test-online-classifier")

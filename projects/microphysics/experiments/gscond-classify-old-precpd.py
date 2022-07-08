import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
    set_prognostic_emulation_model,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-07-01/combined-gscond-precpd-v1/model.tf"  # noqa
CLASSIFIER = "gs://vcm-ml-experiments/microphysics-emulation/2022-07-07/gscond-precpd-classifier-v1/checkpoints/epoch.041.tf"  # noqa

mask_levels = {
    "cloud_water_mixing_ratio_after_gscond": dict(start=74, stop=None),
    "specific_humidity_after_gscond": dict(start=74, stop=None),
    "air_temperature_after_gscond": dict(start=74, stop=None),
    "cloud_water_mixing_ratio_after_precpd": dict(start=74, stop=None),
    "specific_humidity_after_precpd": dict(start=74, stop=None),
    "air_temperature_after_precpd": dict(start=74, stop=None),
}


def _get_job():
    config = load_yaml("../configs/default.yaml")

    config = set_prognostic_emulation_model(
        config,
        MODEL,
        gscond_only=False,
        gscond_cloud_conservative=True,
        classifier_path=CLASSIFIER,
        mask_emulator_levels=mask_levels,
        mask_gscond_zero_cloud_classifier=True,
        mask_precpd_zero_cloud_classifier=True,
        enforce_conservative=True,
    )

    return PrognosticJob(
        name=f"full-zc-emulation-zero-cloud-30d-v1",
        image_tag="1e93473324cf76e807b96baa5ffd0f39d177be1b",
        config=config,
    )


submit_jobs([_get_job()], f"full-zc-emulation-online")

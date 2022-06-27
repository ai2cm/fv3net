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


def _get_job(classify_zero_cloud, classify_zero_tend):
    config = load_yaml("../configs/default.yaml")

    config = set_prognostic_emulation_model(
        config, MODEL, gscond_only=True, gscond_conservative=True
    )
    gscond_config = config["zhao_carr_emulation"]["gscond"]
    gscond_config["classifier_path"] = CLASSIFIER
    gscond_config["mask_emulator_levels"] = mask_levels
    gscond_config["mask_gscond_zero_cloud_classifier"] = classify_zero_cloud
    gscond_config["mask_gscond_zero_tend_classifier"] = classify_zero_tend
    gscond_config["enforce_conservative"] = True

    zcloud_tag = "zcloud-" if classify_zero_cloud else ""
    ztend_tag = "ztend-" if classify_zero_tend else ""

    return PrognosticJob(
        name=f"gscond-only-classifier-{zcloud_tag}{ztend_tag}online-10d-v2",
        image_tag="59317e157f079dee5de07244695c96e6c6f38215",
        config=config,
    )


jobs = [_get_job(zc, zt) for zc, zt in [(False, True), (True, False), (True, True)]]
submit_jobs(jobs, f"test-online-classifier")

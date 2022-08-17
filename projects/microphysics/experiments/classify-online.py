import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1/model.tf"  # noqa
CLASSIFIER = "gs://vcm-ml-experiments/microphysics-emulation/2022-06-09/gscond-classifier-v1/model.tf"  # noqa

mask_levels = {
    "cloud_water_mixing_ratio_after_gscond": dict(start=74, stop=None),
    "specific_humidity_after_gscond": dict(start=74, stop=None),
    "air_temperature_after_gscond": dict(start=74, stop=None),
}


def _get_job(classify_zero_cloud, classify_no_tend):
    config = load_yaml("../configs/default.yaml")

    gscond_config = {
        "path": MODEL,
        "classifier_path": CLASSIFIER,
        "gscond_cloud_conservative": True,
        "mask_emulator_levels": mask_levels,
        "mask_gscond_zero_cloud_classifier": classify_zero_cloud,
        "mask_gscond_no_tend_classifir": classify_no_tend,
        "enforce_conservative": True,
    }

    config["zhao_carr_emulation"] = {"gscond": gscond_config}
    config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = True

    zcloud_tag = "zcloud-" if classify_zero_cloud else ""
    ztend_tag = "ztend-" if classify_no_tend else ""

    return PrognosticJob(
        name=f"gscond-only-classifier-{zcloud_tag}{ztend_tag}online-10d-v3",
        image_tag="f580dfd909dd9e785131b4b34861b2224bc0b34a",
        config=config,
    )


jobs = [_get_job(zc, zt) for zc, zt in [(False, True), (True, False), (True, True)]]
submit_jobs(jobs, f"test-online-classifier")

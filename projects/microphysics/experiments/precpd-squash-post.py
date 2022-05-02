import sys
import itertools

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-29/dqc-precpd-limiter-rnn-limited-qc-87e803/model.tf"  # noqa: E501


def _get_job(conservative_cloud: bool, squash: float):
    config = load_yaml("../configs/default.yaml")

    config["zhao_carr_emulation"] = {
        "model": {
            "path": MODEL,
            "gscond_cloud_conservative": conservative_cloud,
            "cloud_squash": squash,
        }
    }
    config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = False
    cons = "-conser" if conservative_cloud else ""
    return PrognosticJob(
        name=f"precpd-lim-squash-post-v1-{squash:.0e}{cons}",
        image_tag="db3c57138817c74d0be8efb9a52c27018feec85b",
        config=config,
    )


def _gen():
    for args in itertools.product([True, False], [1e-5, 1e-6]):
        yield _get_job(*args)


submit_jobs(list(_gen()), f"precpd-limit-squash-posthoc")

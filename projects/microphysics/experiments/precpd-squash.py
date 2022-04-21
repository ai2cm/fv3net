import sys
import itertools

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/dqc-precpd-limiter-rnn-limited-qc-87e803-30d-v1-online/model.tf"  # noqa: E501


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
    cons = "-conservative" if conservative_cloud else ""
    return PrognosticJob(
        name=f"precpd-limit-squash-v3-{squash:.0e}{cons}",
        image_tag="98ea028bec53447f5bb94c4c3ce5bd330a690d52",
        config=config,
    )


def _gen():
    for args in itertools.product([True, False], [1e-5, 1e-6]):
        yield _get_job(*args)


submit_jobs(list(_gen()), f"precpd-limit-squash")

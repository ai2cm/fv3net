import sys

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-13/gscond-only-dense-local-nfiles1980-41b1c1-v1/model.tf"  # noqa


def _get_job():
    config = load_yaml("../configs/gscond-only.yaml")
    config["zhao_carr_emulation"] = {
        "gscond": {
            "path": MODEL,
            "gscond_cloud_conservative": True,
            "enforce_conservative": True,
        }
    }
    return PrognosticJob(
        name=f"conserve-heat",
        image_tag="1dda99448fed734830884e2217d772710d12ed22",
        config=config,
    )


def _gen():
    yield _get_job()


submit_jobs(list(_gen()), f"conserve-heat")

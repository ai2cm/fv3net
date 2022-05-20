import sys
import itertools

sys.path.insert(0, "../argo")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402

MODEL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-all-loss-rnn-7ef273/model.tf"  # noqa: E501

# set gscond only model


def _get_job(model_type: str, conservative_cloud: bool, squash: float):
    config = load_yaml("../configs/default.yaml")
    emulate_gscond_only = {"gscondonly": True, "full": False}[model_type]
    model_key = "gscond" if emulate_gscond_only else "model"

    config["zhao_carr_emulation"] = {
        model_key: {
            "path": MODEL,
            "gscond_cloud_conservative": conservative_cloud,
            "cloud_squash": squash,
        }
    }
    config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = emulate_gscond_only
    cons = "-conservative" if conservative_cloud else ""
    return PrognosticJob(
        name=f"squash-v2-{model_type}-{squash:.0e}{cons}",
        image_tag="38d0f775ba1baa4a2ba255ac18befd2dccfced2b",
        config=config,
    )


def _gen():
    for args in itertools.product(
        ["gscondonly", "full"], [True, False], [1e-4, 1e-5, 1e-6]
    ):
        yield _get_job(*args)


submit_jobs(list(_gen()), f"squash")

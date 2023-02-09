import sys

sys.path.insert(0, "../scripts")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402


BASE_URL = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2-{month:d}/artifacts/2016{month:02d}01.000000/RESTART"  # noqa


def get_job(month: int):

    config = load_yaml("../configs/gscond-and-precpd.yaml")["config"]
    config["duration"] = "30d"
    config["initial_conditions"] = BASE_URL.format(month=month)

    tag_sha = "95218aec9ed1e6cbda0db1e1f69dc34e0d839c52"
    return PrognosticJob(
        f"zc-emu-monthly-normfix-m{month:02d}-30d-v1",
        config=config,
        image_tag=tag_sha,
        fv3net_image_tag=tag_sha,
    )


jobs = [get_job(i) for i in range(1, 13)]
submit_jobs(jobs, experiment_name="zc-emu-monthly-fix-norm")

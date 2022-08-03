import sys

sys.path.insert(0, "../scripts")

from end_to_end import PrognosticJob, load_yaml, submit_jobs  # noqa: E402


BASE_URL = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2-{month:d}/artifacts/2016{month:02d}01.000000/RESTART"  # noqa


def get_job(month: int):

    config = load_yaml("../configs/gscond-and-precpd.yaml")["config"]
    config["initial_conditions"] = BASE_URL.format(month=month)

    return PrognosticJob(
        f"combined-emu-monthly-init-{month:02d}-v1", config=config, image_tag="latest"
    )


jobs = [get_job(i) for i in range(1, 12)]
submit_jobs(jobs, experiment_name="online-all-month-init")

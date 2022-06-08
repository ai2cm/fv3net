import io
import os

import fv3config
import sys
import vcm
from fv3net.artifacts.metadata import StepMetadata

from runtime.config import get_model_urls
from runtime.segmented_run.run import run_segment


def read_last_segment(run_url):
    fs = vcm.get_fs(run_url)
    artifacts_dir = os.path.join(run_url, "artifacts")
    try:
        segments = sorted(fs.ls(artifacts_dir))
    except FileNotFoundError:
        segments = []

    if len(segments) > 0:
        return vcm.to_url(fs, segments[-1])


def read_run_config(run_url):
    fs = vcm.get_fs(run_url)
    s = fs.cat(os.path.join(run_url, "fv3config.yml"))
    return fv3config.load(io.BytesIO(s))


def append_segment_to_run_url(run_url, ic_date: str):
    """Append an segment to an initialized segmented run

    Either runs the first segment, or additional ones
    """
    dir_ = "."

    config = read_run_config(run_url)
    model_urls = get_model_urls(config)
    StepMetadata(
        job_type="prognostic_run",
        url=run_url,
        dependencies={"ml_models": model_urls} if len(model_urls) > 0 else None,
    ).print_json()

    config = fv3config.enable_restart(
        config,
        os.path.join(
            "gs://vcm-ml-experiments/n2f-pire-stable-ml/2022-06-03",
            "tapering-effect-mae-no-taper-mse-limiter-off-seed-1/fv3gfs_run/artifacts",
            f"{ic_date}.000000/",
            "RESTART",
        ),
    )

    rundir = os.path.join(dir_, f"rundir_ic_{ic_date}_6hr")

    exit_code = run_segment(config, rundir)

    return exit_code


append_segment_to_run_url(".", sys.argv[1])

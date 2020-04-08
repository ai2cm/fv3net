import argparse
import logging
import os
import fsspec
import yaml
import fv3config
from fv3net.pipelines.common import get_alphanumeric_unique_tag
import vcm

logger = logging.getLogger("run_jobs")


def _get_cpu_count_required(config):
    layout = config["namelist"]["fv_core_nml"]["layout"]
    return 6 * layout[0] * layout[1]


def _get_jobname(config):
    experiment_name = config["experiment_name"]
    unique_tag = get_alphanumeric_unique_tag(8)
    return f"{experiment_name}-{unique_tag}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="Path to fv3config yaml.",
    )
    parser.add_argument(
        "outdir", type=str, help="Remote url where output will be saved.",
    )
    parser.add_argument(
        "--dockerimage",
        type=str,
        required=False,
        default="us.gcr.io/vcm-ml/fv3gfs-python",
    )
    parser.add_argument(
        "--runfile", type=str, required=False, default=None,
    )
    args = parser.parse_args()
    with fsspec.open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cpu_count_required = _get_cpu_count_required(config)
    jobname = _get_jobname(config)
    fs = vcm.cloud.get_fs(args.outdir)
    runfile = args.runfile
    if runfile is not None:
        remote_runfile = os.path.join(args.outdir, "config", "runfile.py")
        fs.put(runfile, remote_runfile)
        runfile = remote_runfile
    fv3config.run_kubernetes(
        args.config,
        args.outdir,
        args.dockerimage,
        runfile=runfile,
        jobname=jobname,
        memory_gb=15,
        cpu_count=cpu_count_required,
    )
    logger.info(f"Submitted {jobname}")

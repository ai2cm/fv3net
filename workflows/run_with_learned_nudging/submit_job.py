import argparse
import logging
import os
import uuid

import fsspec
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

DOCKER_IMAGE = "us.gcr.io/vcm-ml/learn-nudging"
RUNFILE = "learn_nudging_runfile.py"
DEFAULT_CONFIG_YAML = "gs://vcm-ml-data/2020-01-29-baseline-FV3GFS-runs/free-2016-C48-npz63/config/fv3config.yml"
DEFAULT_OUTDIR = "local_outdir"


def submit_job(outdir, config, run_local):
    config["namelist"]["coupler_nml"]["days"] = 91
    config["namelist"]["fv_core_nml"]["layout"] = [1,1]
    if run_local:
        fv3config.run_docker(
            config,
            outdir,
            DOCKER_IMAGE,
            runfile=RUNFILE,
        )
    else:
        config_dir = os.path.join(outdir, "config")
        remote_config_location = os.path.join(config_dir, "fv3config.yml")
        remote_runfile_location = os.path.join(config_dir, "runfile.py")
        with fsspec.open(remote_config_location, "w") as f:
            f.write(yaml.dump(config))
        fs, _, _ = fsspec.get_fs_token_paths(remote_runfile_location)
        fs.put(RUNFILE, remote_runfile_location)
        fv3config.run_kubernetes(
            remote_config_location,
            os.path.join(outdir, "output"),
            DOCKER_IMAGE,
            runfile=remote_runfile_location,
            jobname="2016.nudged-to-monthly-mean." + str(uuid.uuid4())[:8],
            memory_gb=15,
            cpu_count=6,
            gcp_secret="gcp-key",
            image_pull_policy="Always",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Path where output from model run will be saved",
    )
    parser.add_argument(
        "--run-yaml",
        type=str,
        default=DEFAULT_CONFIG_YAML,
        help="Path to fv3config yaml.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Do run locally use run_docker()"
    )
    args = parser.parse_args()
    with fsspec.open(args.run_yaml) as file:
        run_config = yaml.safe_load(file)
    submit_job(args.outdir, run_config, args.local)

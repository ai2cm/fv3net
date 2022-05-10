import argparse
import sys
import glob
import os

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402


def _get_job(config: str, revision: str, suffix: str, test: bool):
    config_name = os.path.splitext(os.path.basename(config))[0]
    config = load_yaml(config)

    if test:
        config["nfiles"] = 50
        config["epochs"] = 3
        config["nfiles_valid"] = 10
        run_config = "default_short"
    else:
        run_config = "default"

    return EndToEndJob(
        name=f"{config_name}-{revision[:6]}-{suffix}",
        fv3fit_image_tag=revision,
        image_tag=revision,
        ml_config=config,
        prog_config=load_yaml(f"../configs/{run_config}.yaml"),
    )


configs = glob.glob("../train/*.yaml")
parser = argparse.ArgumentParser()
parser.add_argument("--revision", default="latest")
parser.add_argument("--suffix", default="v1")
parser.add_argument("--test", action="store_true")

args = parser.parse_args()
jobs = [_get_job(config, args.revision, args.suffix, args.test) for config in configs]
test_str = "test-" if args.test else ""
submit_jobs(jobs, f"emu-end-to-end-{test_str}{args.revision[:6]}")

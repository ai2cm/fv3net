import sys
import glob
import os

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402


def _get_job(config: str, revision):
    config_name = os.path.splitext(os.path.basename(config))[0]
    config = load_yaml(config)
    return EndToEndJob(
        name=f"{config_name}-{revision[:6]}-{suffix}",
        fv3fit_image_tag=revision,
        image_tag=revision,
        ml_config=config,
        prog_config=load_yaml("../configs/default.yaml"),
    )


configs = glob.glob("../train/*.yaml")

revision = sys.argv[1] if len(sys.argv) > 1 else "latest"
suffix = sys.argv[2]
jobs = [_get_job(config, revision) for config in configs]
submit_jobs(jobs, f"{revision}-end-to-end")

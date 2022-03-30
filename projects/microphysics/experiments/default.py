import sys

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402


def _get_job(config_name: str, revision):
    config = load_yaml(f"../train/{config_name}.yaml")
    return EndToEndJob(
        name=f"{config_name}-{revision[:6]}-v1",
        fv3fit_image_tag=revision,
        image_tag=revision,
        ml_config=config,
        prog_config=load_yaml("../configs/default_short.yaml"),
    )


revision = sys.argv[1] if len(sys.argv) > 1 else "latest"
jobs = [_get_job(config, revision) for config in ["limited", "rnn"]]
submit_jobs(jobs, f"{revision}-end-to-end")

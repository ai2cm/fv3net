import sys

sys.path.insert(0, "../scripts")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


dense_like = {"width": 256, "depth": 2}
rnn = {"channels": 256, "depth": 2}
arch_params = {
    "dense": dense_like,
    "rnn-v1-shared-weights": rnn,
}


def _get_job(arch_key: str, revision: str):
    train_config = load_yaml(f"../train/rnn-precpd-only.yaml")
    train_config["model"]["architecture"]["name"] = arch_key
    train_config["model"]["architecture"]["kwargs"] = arch_params[arch_key]

    return TrainingJob(
        name=f"precpd-diff-only-{arch_key}-v1",
        config=train_config,
        fv3fit_image_tag=revision,
    )


arch_keys = ["dense", "rnn-v1-shared-weights"]
revision = "latest"
jobs = [_get_job(arch_key, revision) for arch_key in arch_keys]

submit_jobs(jobs, f"precpd-diff-only-July2022")

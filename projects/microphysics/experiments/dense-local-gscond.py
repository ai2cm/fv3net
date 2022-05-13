import argparse
import sys
from itertools import product

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402


dense_like = {"width": 256, "depth": 2}
rnn = {"channels": 256, "depth": 2}
hybrid = {"channels": 256, "dense_depth": 1, "dense_width": 256}
arch_params = {
    "dense": dense_like,
    "dense-local": dense_like,
    "rnn": hybrid,
    "rnn-v1-shared-weights": rnn,
}

# It's a PITA to adjust transforms programmatically at this point time
# If you want to get the config for non-tscaled experiments
# https://github.com/ai2cm/fv3net/blob/74d20d5fd78d6fe73f85d82f81a828e9324a5b24/projects/microphysics/train/gscond-only.yaml # noqa


def _get_job(config_name: str, arch_key: str, revision: str, suffix: str):
    train_config = load_yaml(f"../train/gscond-only.yaml")
    train_config["model"]["architecture"]["name"] = arch_key
    train_config["model"]["architecture"]["kwargs"] = arch_params[arch_key]

    prog_config = load_yaml(f"../configs/default.yaml")
    prog_config["duration"] = "2d"

    return EndToEndJob(
        name=f"{config_name}-{arch_key}-{revision[:6]}-{suffix}",
        fv3fit_image_tag=revision,
        image_tag=revision,
        ml_config=train_config,
        prog_config=prog_config,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--revision", default="latest")
parser.add_argument("--suffix", default="v1")

arch_keys = ["dense-local", "rnn-v1-shared-weights"]
configs = ["gscond-only"]
args = parser.parse_args()

jobs = [
    _get_job(config_name, arch_key, args.revision, args.suffix)
    for config_name, arch_key in product(configs, arch_keys)
]
submit_jobs(jobs, f"gscond-only-emu-may2022")

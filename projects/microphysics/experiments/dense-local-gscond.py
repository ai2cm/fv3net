import argparse
import sys

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


def _get_job(arch_key: str, revision: str, suffix: str):
    train_config = load_yaml("../train/gscond-only.yaml")
    train_config["model"]["architecture"]["name"] = arch_key
    train_config["model"]["architecture"]["kwargs"] = arch_params[arch_key]

    prog_config = load_yaml(f"../configs/default.yaml")
    prog_config["duration"] = "2d"
    prog_config["namelist"]["gfs_physics_nml"]["emulate_gscond_only"] = True

    return EndToEndJob(
        name=f"gscond-only-{arch_key}-{revision[:6]}-{suffix}",
        fv3fit_image_tag=revision,
        image_tag=revision,
        ml_config=train_config,
        prog_config=prog_config,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--revision", default="latest")
parser.add_argument("--suffix", default="v1")

arch_keys = ["dense", "dense-local", "rnn", "rnn-v1-shared-weights"]
args = parser.parse_args()
jobs = [_get_job(arch_key, args.revision, args.suffix) for arch_key in arch_keys]
submit_jobs(jobs, f"gscond-only-emu-may2022")

import sys
import secrets


sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402

group = secrets.token_hex(3)


def _get_job(config_name: str, exp_tag: str, limit_negative_qc: bool):
    config = load_yaml(f"../train/{config_name}.yaml")

    if not limit_negative_qc:
        # Tensor transforms are hard to edit programmatically
        # Do a simple check to error if things change
        key = "cloud_water_mixing_ratio_after_precpd"
        assert config.tensor_transforms[0]["source"] == key
        assert config.tensor_transforms[0]["to"] == key

        del config.tensor_transforms[0]

    return EndToEndJob(
        name=f"precpd-limiting-{exp_tag}-{group}",
        fv3fit_image_tag="4a363b9600e3092a7a050248eb96fc1926fcf49a",
        image_tag="4a363b9600e3092a7a050248eb96fc1926fcf49a",
        ml_config=config,
        prog_config=load_yaml("../configs/default.yaml"),
    )


jobs = [
    _get_job("rnn-no-limit", "rnn-control", limit_negative_qc=False),
    _get_job("rnn", "rnn-limit-precpd-tend", limit_negative_qc=False),
    _get_job("rnn", "rnn-limit-precpd-tend-limit-qc", limit_negative_qc=True),
]
submit_jobs(jobs, f"dqc-precpd-limiting")

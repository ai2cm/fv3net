import sys
import secrets


sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402

group = secrets.token_hex(3)


def _get_job(
    config_name: str,
    exp_tag: str,
    limit_negative_qc: bool = False,
    limit_negative_qv: bool = False,
):
    config = load_yaml(f"../train/{config_name}.yaml")

    if not limit_negative_qc:
        # Tensor transforms are hard to edit programmatically
        # Do a simple check to error if things change
        key = "cloud_water_mixing_ratio_after_precpd"
        assert config["tensor_transform"][0]["source"] == key
        assert config["tensor_transform"][0]["to"] == key

        del config["tensor_transform"][0]

    if limit_negative_qv:
        key = "specific_humidity_after_precpd"
        config["tensor_transform"].insert(
            0, dict(source=key, to=key, transform=dict(lower=0.0))
        )

    return EndToEndJob(
        name=f"precpd-limiting-{exp_tag}-{group}",
        fv3fit_image_tag="82654b2321ac6f4dc2fdc743588ae335598982e0",
        image_tag="82654b2321ac6f4dc2fdc743588ae335598982e0",
        ml_config=config,
        prog_config=load_yaml("../configs/default.yaml"),
    )


jobs = [
    _get_job("rnn-no-limiting", "rnn-control"),
    _get_job("rnn", "rnn-limit-precpd-tend"),
    _get_job("rnn", "rnn-limit-precpd-tend-limit-qc", limit_negative_qc=True),
    _get_job(
        "rnn",
        "rnn-limit-precpd-tend-limit-qc-qv",
        limit_negative_qc=True,
        limit_negative_qv=True,
    ),
]
submit_jobs(jobs, f"dqc-precpd-limiting")

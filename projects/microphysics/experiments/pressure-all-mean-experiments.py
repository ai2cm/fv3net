import sys

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402


def control(ml_config):
    pass


def norm_air_pressure_all(ml_config):
    ml_config["model"]["normalize_map"] = {
        "air_pressure": {"center": "all", "scale": "all"}
    }


def norm_all_all(ml_config):
    ml_config["model"]["normalize_default"] = {"center": "all", "scale": "all"}


def _get_job(config_name: str, norm: str):
    config = load_yaml(f"../train/{config_name}.yaml")
    norm_func = globals()[norm]
    norm_func(config)
    argo_friendly_name = norm.replace("_", "-")
    return EndToEndJob(
        name=f"{config_name}-{argo_friendly_name}-v1",
        fv3fit_image_tag="d848e586db85108eb142863e600741621307502b",
        image_tag="d848e586db85108eb142863e600741621307502b",
        ml_config=config,
        prog_config=load_yaml("../configs/default_short.yaml"),
    )


jobs = [
    _get_job(config, norm)
    for config in ["limited", "rnn"]
    for norm in ["control", "norm_all_all", "norm_air_pressure_all"]
]

submit_jobs(jobs, "more-normalization")

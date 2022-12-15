import sys

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402

NAME = "zc-emu-precip-{}-v1"


def get_job(simple, strict):
    config = load_yaml("../configs/combined-base.yaml")

    config["duration"] = "30d"
    model_cfg = config["zhao_carr_emulation"]["model"]
    model_cfg["simple_precip_conservative"] = simple
    model_cfg["enforce_strict_precpd_conservative"] = strict

    if simple:
        name = NAME.format("simple-conserve")
    elif strict:
        name = NAME.format("strict-conserve")
    else:
        name = NAME.format("no-conserve")

    return PrognosticJob(
        name=name, image_tag="99b74dcc46b233c05942b32098c1207b0e6dd323", config=config,
    )


jobs = [get_job(*flags) for flags in [[True, False], [False, True], [False, False]]]
submit_jobs(jobs, "compare-precip-conservation-dec2022")

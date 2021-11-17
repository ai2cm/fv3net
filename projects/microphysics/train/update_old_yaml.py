"""A script to update yamls saved from around October 2021 to the latest
interface.
"""
import dataclasses
import sys

import dacite
import yaml
from fv3fit.train_microphysics import SliceConfig, TrainConfig

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)


selections = {}
for key, val in config.get("model", {}).pop("selection_map", {}).items():
    selections[key] = SliceConfig(*val)
if selections:
    config["model"]["selection_map"] = selections

config["loss"] = {}
for key in ["loss_variables", "optimizer", "weights", "metric_variables"]:
    try:
        config["loss"][key] = config.pop(key)
    except KeyError:
        pass

config["wandb"] = {
    "wandb_project": config.pop("wandb_project", "microphysics-emulation"),
    "job_type": "training",
}

config.pop("wandb_model_name", None)

loaded = dacite.from_dict(TrainConfig, config, config=dacite.Config(strict=True))
yaml.safe_dump(dataclasses.asdict(loaded), sys.stdout)

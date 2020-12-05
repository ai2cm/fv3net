from typing import Dict
import yaml
import f90nml

FV3CONFIG_FILENAME = "fv3config.yml"


def get_config() -> Dict:
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return config


def get_namelist() -> f90nml.Namelist:
    return f90nml.read("input.nml")

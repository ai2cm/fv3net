from typing import Dict
import yaml
import f90nml
import fv3config

FV3CONFIG_FILENAME = "fv3config.yml"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_config() -> Dict:
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return config


def get_namelist() -> f90nml.Namelist:
    return f90nml.read("input.nml")

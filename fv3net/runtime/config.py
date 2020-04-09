import yaml
import f90nml
import fv3config

FV3CONFIG_FILENAME = "fv3config.yml"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_runfile_config():
    with open(FV3CONFIG_FILENAME) as f:
        config = yaml.safe_load(f)
    return dotdict(config["scikit_learn"])


def get_config():
    """Return fv3config dictionary"""
    return fv3config.config_from_yaml(FV3CONFIG_FILENAME)


def get_namelist():
    return f90nml.read("input.nml")

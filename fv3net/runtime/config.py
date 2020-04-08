import yaml
import f90nml

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


# TODO: refactor to use fv3config.config_from_yaml
def get_config():
    """Return fv3config dictionary"""
    with open(FV3CONFIG_FILENAME) as f:
        config = yaml.safe_load(f)
    return config


def get_namelist():
    return f90nml.read("input.nml")


# TODO: delete and replace usages with fv3config.get_timestep()
def get_timestep():
    """Return model timestep in seconds"""
    return get_namelist()["coupler_nml"]["dt_atmos"]

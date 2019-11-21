import os

from .docker import make_experiment


def inside_docker():
    return os.path.exists("/FV3/fv3.exe")


if inside_docker():
    from .native import run_experiment
else:
    from .docker import run_experiment

import os

from .docker import make_experiment

_inside_docker = os.path.exists("/FV3/fv3.exe")

if _inside_docker:
    from .native import run_experiment
else:
    from .docker import run_experiment

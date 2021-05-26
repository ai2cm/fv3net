import sys
from fv3net.artifacts.query import get_artifacts, Step, main
import pathlib

import pytest


local_experiment_root = (
    (pathlib.Path(__file__).parent / "experiment-root").absolute().as_posix()
)


@pytest.mark.parametrize("exp_root", [local_experiment_root, "gs://vcm-ml-experiments"])
def test_get_runs(exp_root):
    steps = get_artifacts(exp_root, ["default"])

    assert steps
    for step in steps:
        assert isinstance(step, Step)


def test_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["artifacts", "--bucket", local_experiment_root])
    main()

import sys
from fv3net.artifacts.query import get_artifacts, Step, main, matches_tag
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
    monkeypatch.setattr(
        sys, "argv", ["artifacts", "ls", "--bucket", local_experiment_root]
    )
    main()


@pytest.mark.parametrize(
    "actual_tag, searched_tag, expected",
    [
        ("some-run", "some-run-1", False),
        ("superstring", "string", True),
        ("one", "two", False),
        ("one", None, True),
    ],
)
def test_matches_tag(actual_tag, searched_tag, expected):
    out = Step(
        None,
        path=None,
        date_created=None,
        project="default",
        tag=actual_tag,
        step="fv3gfs_run",
    )
    assert matches_tag(out, searched_tag) == expected

import pathlib
import sys
from fv3net.artifacts.cli import main


local_experiment_root = (
    (pathlib.Path(__file__).parent / "experiment-root").absolute().as_posix()
)


def test_cli(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["artifacts", "ls", "--bucket", local_experiment_root]
    )
    main()

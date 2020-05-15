import synth
from pathlib import Path


def open_schema(localpath):
    path = Path(__file__)
    diag_path = path.parent / localpath
    with open(diag_path) as f:
        return synth.generate(synth.load(f))


def test_run():

    diag_schema = open_schema("diag.json")
    restart_schema = open_schema("restart.json")

    assert False

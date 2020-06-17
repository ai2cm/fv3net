import pytest
from generate_samples import main
import itertools


def _no_repeated_values(seq):
    return len(seq) == len(set(seq))


@pytest.mark.parametrize(
    "non_timestep, repeated_timestep", itertools.product([True, False], [True, False])
)
def test_generate_samples(tmpdir, non_timestep, repeated_timestep):
    if non_timestep:
        tmpdir.join("not_a_timestep").write("")

    if repeated_timestep:
        tmpdir.join("20160802.003000.tile1.nc").write("")
        tmpdir.join("20160802.003000.tile2.nc").write("")

    tmpdir.mkdir("20160801.000000")
    tmpdir.mkdir("20160801.001500")
    tmpdir.mkdir("20160801.003000")
    tmpdir.mkdir("20160801.004500")
    tmpdir.mkdir("20160801.010000")
    tmpdir.mkdir("20160801.011500")
    tmpdir.mkdir("20160801.013000")
    tmpdir.mkdir("20160801.014500")
    tmpdir.mkdir("20160801.020000")

    args = {
        "url": str(tmpdir),
        "spinup": "20160801.001500",
        "boundary": "20160801.013000",
        "train_samples": 2,
        "test_samples": 2,
        "seed": "0",
    }
    output = main(args)

    assert len(output["train_and_test"]["train"]) == args["train_samples"]
    assert len(output["train_and_test"]["test"]) == args["test_samples"]
    assert _no_repeated_values(output["one_step"])


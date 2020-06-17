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

    tmpdir.mkdir("20160801.000000")
    tmpdir.mkdir("20160801.001500")
    # spinup period done
    tmpdir.mkdir("20160801.003000")
    tmpdir.mkdir("20160801.004500")
    # begin test period
    if repeated_timestep:
        tmpdir.join("20160801.001000.tile1.nc").write("")
        tmpdir.join("20160801.001000.tile2.nc").write("")
    else:
        tmpdir.mkdir("20160801.010000")
    tmpdir.mkdir("20160801.011500")

    args = {
        "url": str(tmpdir),
        "spinup": "20160801.001500",
        "boundary": "20160801.004500",
        "train_samples": 1,
        "test_samples": 1,
        "seed": "0",
    }
    output = main(args)

    assert len(output["train_and_test"]["train"]) == args["train_samples"]
    assert len(output["train_and_test"]["test"]) == args["test_samples"]
    assert _no_repeated_values(output["one_step"])


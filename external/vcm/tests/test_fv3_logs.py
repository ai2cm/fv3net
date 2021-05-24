import datetime
from vcm.fv3.logs import FV3Log, concatenate, loads
from pathlib import Path
import pytest
import yaml


@pytest.fixture(params=["shield", "fv3net"], scope="session")
def shield_log(request):
    if request.param == "shield":
        path = (
            Path(__file__).parent
            / "C3072_0801_n11a_vulcan-2020-05-21-production-simulation.o268642462"
        )
        with open(path) as f:
            str_ = f.read()
        return str_
    else:
        path = Path(__file__).parent / "prognostic-run-logs.txt"
        with open(path) as f:
            str_ = f.read()
        return str_


def test_loads_with_shield_data(shield_log: str):
    fv3log = loads(shield_log)

    for key in fv3log.totals:
        assert len(fv3log.totals[key]) == len(fv3log.dates)


def test_loads_with_shield_data_date(shield_log: str, regtest):
    fv3log = loads(shield_log)
    yaml.safe_dump(fv3log.dates, regtest)


@pytest.mark.parametrize(
    "variable_name",
    [
        "total surface pressure",
        "total water vapor",
        "mean dry surface pressure",
        "total cloud water",
        "total rain water",
        "total cloud ice",
        "total snow",
        "total graupel",
    ],
)
def test_loads_shield_totals(shield_log: str, regtest, variable_name: str):
    fv3log = loads(shield_log)
    yaml.safe_dump(fv3log.totals[variable_name], regtest)


def test_concatenate():
    log1 = FV3Log(
        dates=[datetime.datetime(2016, 1, 1)], totals={"a": [1.0]}, ranges={"a": (0, 1)}
    )
    log2 = FV3Log(
        dates=[datetime.datetime(2016, 1, 2)], totals={"a": [1.0]}, ranges={"a": (0, 1)}
    )

    log = concatenate([log1, log2])

    assert log.dates == log1.dates + log2.dates
    assert log.totals["a"] == log1.totals["a"] + log2.totals["a"]
    assert log.ranges["a"] == log1.ranges["a"] + log2.ranges["a"]

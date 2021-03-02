from vcm.fv3.logs import loads, parse_date_forecast_date
from datetime import datetime
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


def test_parse_date():
    example_date = "0 FORECAST DATE          14 AUG.  2016 AT 18 HRS  0.00 MINS"
    assert parse_date_forecast_date(example_date) == datetime(
        year=2016, month=8, day=14, hour=18
    )


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

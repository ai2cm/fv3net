from datetime import datetime
import numpy as np
from gallery import __version__, plot_daily_and_hourly_hist
import pytest


def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    "timesteps",
    [
        [datetime(2016, 8, 1), datetime(2016, 8, 2)],
        (datetime(2016, 8, 1), datetime(2016, 8, 2)),
        [np.datetime64("2016-08-01"), np.datetime64("2016-08-02")],
        np.array(["2016-08-01", "2016-08-02"], dtype="datetime64"),
    ],
)
def test_plot_daily_and_hourly_hist(timesteps):
    plot_daily_and_hourly_hist(timesteps)

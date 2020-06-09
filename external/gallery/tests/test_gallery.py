from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import cftime

import pytest
from gallery import (
    __version__,
    plot_daily_and_hourly_hist,
    plot_daily_hist,
    plot_hourly_hist,
)


def test_version():
    assert __version__ == "0.1.0"


example_timesteps = [
    [datetime(2016, 8, 1), datetime(2016, 8, 2)],
    (datetime(2016, 8, 1), datetime(2016, 8, 2)),
    [np.datetime64("2016-08-01"), np.datetime64("2016-08-02")],
    np.array(["2016-08-01", "2016-08-02"], dtype="datetime64"),
]


@pytest.mark.parametrize("timesteps", example_timesteps)
def test_plot_daily_and_hourly_hist(timesteps):
    plot_daily_and_hourly_hist(timesteps)


@pytest.mark.parametrize("timesteps", example_timesteps)
def test_plot_daily_hist(timesteps):
    plot_daily_hist(plt.subplot(), timesteps)


@pytest.mark.parametrize("timesteps", example_timesteps)
def test_plot_hourly_hist(timesteps):
    plot_hourly_hist(plt.subplot(), timesteps)

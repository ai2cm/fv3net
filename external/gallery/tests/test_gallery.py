from datetime import datetime
from gallery import __version__, plot_daily_and_hourly_hist


def test_version():
    assert __version__ == "0.1.0"


def test_plot_daily_and_hourly_hist():
    timesteps = [datetime(2016, 8, 1), datetime(2016, 8, 2)]
    plot_daily_and_hourly_hist(timesteps)

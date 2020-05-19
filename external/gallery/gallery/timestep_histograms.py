import datetime
from typing import Sequence, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_daily_and_hourly_hist(
    time_list: Sequence[Union[datetime.datetime, np.datetime64]]
) -> plt.figure:
    """Given a sequence of datetime-like objects, create and return 2-subplot figure with
    histograms of daily and hourly counts."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    plot_daily_hist(axes[0], time_list)
    plot_hourly_hist(axes[1], time_list)
    fig.suptitle(f"total count: {len(time_list)}")
    plt.tight_layout()
    return fig


def plot_daily_hist(ax: Axes, time_list: Sequence[datetime.datetime]):
    """Given list of datetimes, plot histogram of count per calendar day on ax"""
    ser = pd.Series(time_list)
    groupby_list = [ser.dt.year, ser.dt.month, ser.dt.day]
    ser.groupby(groupby_list).count().plot(ax=ax, kind="bar", title=f"Daily count")
    ax.set_ylabel("Count")


def plot_hourly_hist(ax: Axes, time_list: Sequence[datetime.datetime]):
    """Given list of datetimes, plot histogram of count per UTC hour on ax"""
    ser = pd.Series(time_list)
    ser.groupby(ser.dt.hour).count().plot(ax=ax, kind="bar", title="Hourly count")
    ax.set_ylabel("Count")
    ax.set_xlabel("UTC hour")

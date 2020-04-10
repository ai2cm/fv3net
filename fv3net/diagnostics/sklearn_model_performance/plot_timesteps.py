import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_timestep_counts(train_times, test_times, output_dir):
    report_sections = {}
    _plot_daily_hourly_count(train_times, "Training data").savefig(
        os.path.join(output_dir, f"count_of_training_times_used.png")
    )
    _plot_daily_hourly_count(test_times, "Test data").savefig(
        os.path.join(output_dir, f"count_of_test_times_used.png")
    )
    report_sections["Temporal distribution of training and test data"] = [
        "count_of_training_times_used.png",
        "count_of_test_times_used.png",
    ]
    return report_sections


def _plot_daily_hourly_count(time_list, suptitle):
    """Given iterable of datetimes, create figure with plots of daily count
    and hourly count."""
    # pandas can't handle cftime DatetimeJulian, so ensure using python datetime
    time_list = [_ensure_datetime(t) for t in time_list]
    se = pd.Series(time_list)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    se.groupby([se.dt.year, se.dt.month, se.dt.day]).count().plot(
        ax=axes[0], kind="bar", title=f"Daily count (total={len(se)})"
    )
    se.groupby(se.dt.hour).count().plot(ax=axes[1], kind="bar", title="Hourly count")
    axes[0].set_ylabel("Count")
    axes[1].set_xlabel("UTC hour")
    axes[1].set_ylabel("Count")
    fig.suptitle(suptitle)
    return fig


def _ensure_datetime(t):
    return datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)

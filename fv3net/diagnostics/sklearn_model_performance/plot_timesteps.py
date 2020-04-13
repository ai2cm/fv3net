import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


TRAINING_FIG_FILENAME = "count_of_training_times_used.png"
TESTING_FIG_FILENAME = "count_of_test_times_used.png"


def plot_timestep_counts(train_times, test_times, output_dir, dpi_figures):
    """Create and save plots of distribution of training and test timesteps"""
    plot_daily_and_hourly_hist(train_times, "Training data").savefig(
        os.path.join(output_dir, TRAINING_FIG_FILENAME),
        dpi=dpi_figures["timestep_histogram"],
    )
    plot_daily_and_hourly_hist(test_times, "Test data").savefig(
        os.path.join(output_dir, TESTING_FIG_FILENAME),
        dpi=dpi_figures["timestep_histogram"],
    )
    report_sections = {
        "Temporal distribution of training and test data": [
            TRAINING_FIG_FILENAME,
            TESTING_FIG_FILENAME,
        ]
    }
    return report_sections


def plot_daily_and_hourly_hist(time_list, suptitle):
    """Given a list of datetimes, create and return 2-subplot figure with
    histograms of daily and hourly counts.."""
    # pandas can't handle cftime.DatetimeJulian, so ensure time_list is datetimes
    time_list = [_ensure_datetime(t) for t in time_list]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    plot_year_month_day_hist(axes[0], time_list)
    plot_hourly_hist(axes[1], time_list)
    fig.suptitle(f"{suptitle} (total={len(time_list)})")
    plt.tight_layout()
    return fig


def plot_year_month_day_hist(ax, time_list):
    """Given list of datetimes, plot histogram of count per calendar day on ax"""
    ser = pd.Series(time_list)
    groupby_list = [ser.dt.year, ser.dt.month, ser.dt.day]
    ser.groupby(groupby_list).count().plot(ax=ax, kind="bar", title=f"Daily count")
    ax.set_ylabel("Count")


def plot_hourly_hist(ax, time_list):
    """Given list of datetimes, plot histogram of count per UTC hour on ax"""
    ser = pd.Series(time_list)
    ser.groupby(ser.dt.hour).count().plot(ax=ax, kind="bar", title="Hourly count")
    ax.set_ylabel("Count")
    ax.set_xlabel("UTC hour")


def _ensure_datetime(t):
    return datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)

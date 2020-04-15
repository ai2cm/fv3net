import os
import gallery


TESTING_FIG_FILENAME = "count_of_test_times_used.png"


def plot_timestep_counts(timesteps, output_dir, dpi_figures):
    """Create and save plots of distribution of training and test timesteps"""
    gallery.plot_daily_and_hourly_hist(timesteps).savefig(
        os.path.join(output_dir, TESTING_FIG_FILENAME),
        dpi=dpi_figures["timestep_histogram"],
    )
    return {"Time distribution of training samples": [TESTING_FIG_FILENAME]}


import os
import yaml
import gallery

TIMESTEPS_FIG_FILENAME = "count_of_test_times_used.png"


def plot_timestep_counts(output_dir, timesteps_filename, dpi_figures):
    """Create and save plots of distribution of training and test timesteps"""
    timesteps_path = os.path.join(output_dir, timesteps_filename)
    with open(timesteps_path) as f:
        timesteps = yaml.safe_load(f)
    gallery.plot_daily_and_hourly_hist(timesteps).savefig(
        os.path.join(output_dir, TIMESTEPS_FIG_FILENAME),
        dpi=dpi_figures["timestep_histogram"],
    )
    return {"Time distribution of testing samples": [TIMESTEPS_FIG_FILENAME]}

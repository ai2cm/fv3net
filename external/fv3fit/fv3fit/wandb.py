import wandb
import matplotlib.pyplot as plt
import numpy as np

from .tensorboard import plot_to_image


# TODO: currently coupled to the scaling

UNITS = {
    "total_precipitation": "mm / day",
    "specific_humidity_output": "g/kg",
    "tendency_of_specific_humidity_due_to_microphysics": "g/kg/day",
    "cloud_water_mixing_ratio_output": "g/kg",
    "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": "g/kg/day",
    "air_temperature_output": "K",
    "tendency_of_air_temperature_due_to_microphysics": "K/day",
}


def _log_profiles(targ, pred, name):

    for i in range(targ.shape[0]):
        levs = np.arange(targ.shape[1])
        fig = plt.figure()
        fig.set_size_inches(3, 5)
        fig.set_dpi(80)
        plt.plot(targ[i], levs, label="target")
        plt.plot(pred[i], levs, label="prediction")
        plt.title(f"Sample {i+1}: {name}")
        plt.xlabel(f"{UNITS[name]}")
        plt.ylabel("Level")
        wandb.log({f"{name}_sample_{i}": wandb.Image(plot_to_image(fig))})
        plt.close()


def log_sample_profiles(targets, predictions, names, nsamples=4):

    for targ, pred, name in zip(targets, predictions, names):

        targ = targ[:nsamples]
        pred = pred[:nsamples]

        if targ.ndim == 2:
            _log_profiles(targ, pred, name)
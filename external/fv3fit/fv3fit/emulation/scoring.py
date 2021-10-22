import logging
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple, Union


SCALE_VALUES = {
    "air_temperature_output": 1,
    "specific_humidity_output": 1000,  # g / kg
    "cloud_water_mixing_ratio_output": 1000,  # g / kg
    "total_precipitation": 1000 / (900 / (3600 * 24)),  # mm / day
    "tendency_of_air_temperature_due_to_microphysics": 3600 * 24,  # K / day,
    "tendency_of_specific_humidity_due_to_microphysics": 1000
    * (3600 * 24),  # g / kg / day
    "tendency_of_cloud_water_mixing_ratio_due_to_microphysics": 1000
    * (3600 * 24),  # g / kg / day
}


def scale(names, values):

    scaled = []
    for name, value in zip(names, values):
        value *= SCALE_VALUES[name]
        scaled.append(value)

    return scaled


def score(target, prediction):

    bias_all = target - prediction
    se = bias_all ** 2

    mse = np.mean(se).astype(np.float)
    bias = np.mean(bias_all).astype(np.float)

    metrics = {
        "mse": mse,
        "bias": bias,
    }

    if target.ndim == 2 and target.shape[1] > 1:
        mse_prof = np.mean(se, axis=0)
        bias_prof = np.mean(bias_all, axis=0)
        rmse_prof = np.sqrt(mse_prof)

        profiles = {
            "mse_profile": mse_prof,
            "bias_profile": bias_prof,
            "rmse_profile": rmse_prof,
        }
    else:
        profiles = {}

    return metrics, profiles


def score_single_output(target, prediction, name, rescale=True):

    if rescale:
        if name not in SCALE_VALUES:
            pass
        target *= SCALE_VALUES[name]
        prediction *= SCALE_VALUES[name]

    scores, profiles = score(target, prediction)
    # Keys use directory style for wandb chart grouping
    flat_score = {f"{k}/{name}": v for k, v in scores.items()}
    flat_profile = {f"{k}/{name}": v for k, v in profiles.items()}

    return flat_score, flat_profile


def score_multi_output(targets, predictions, names, rescale=True):

    all_scores = {}
    all_profiles = {}

    for target, pred, name in zip(targets, predictions, names):

        scores, profiles = score(target, pred)
        flat_score = {f"{k}/{name}": v for k, v in scores.items()}
        flat_profile = {f"{k}/{name}": v for k, v in profiles.items()}
        all_scores.update(flat_score)
        all_profiles.update(flat_profile)

    # assumes all profiles are same size
    profile = next(iter(all_profiles.values()))
    all_profiles["level"] = np.arange(len(profile))

    return all_scores, all_profiles


def score_model(
    model: tf.keras.Model,
    inputs: Union[tf.Tensor, Tuple[tf.Tensor]],
    targets: Union[tf.Tensor, Sequence[tf.Tensor]],
):





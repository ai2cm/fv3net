import logging
from typing import Sequence, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

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

ScoringOutput = Tuple[Dict[str, float], Dict[str, np.ndarray]]


def score(target, prediction) -> ScoringOutput:
    """
    Calculate overall bias and MSE metrics as well as per-feature
    (e.g., vertical profiles) of bias, MSE, and RMSE for the
    prediction. Return each as a dict of scores.
    """
    bias_all = prediction - target
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


def score_single_output(
    target: np.ndarray, prediction: np.ndarray, name: str, rescale: bool = True
) -> ScoringOutput:
    """
    Score a single named output from an emulation model. Returns
    a flat dictionary with score/profile keys preceding variable
    name for Weights & Biases chart grouping.

    Args:
        target: truth values
        prediction: emulated values
        name: name of field being emulated to be added to the key
            of each score
        rescale: rescale outputs using factor in top-level
            SCALE_VALUES mapping if available
    """

    if rescale:
        try:
            target *= SCALE_VALUES[name]
            prediction *= SCALE_VALUES[name]
        except KeyError:
            logger.error(f"No scaling value found for {name}. Leaving unscaled...")

    scores, profiles = score(target, prediction)
    # Keys use directory style for wandb chart grouping
    flat_score = {f"{k}/{name}": v for k, v in scores.items()}
    flat_profile = {f"{k}/{name}": v for k, v in profiles.items()}

    return flat_score, flat_profile


def score_multi_output(
    targets: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    names: Sequence[str],
    rescale: bool = True,
) -> ScoringOutput:

    """
    Score a multi-output model while retaining a flat dictionary
    structure of the output scores.

    Args:
        target: truth values for each model output
        prediction: emulated values for each model output
        names: names of fields being emulated, used for score key
            uniqueness
        rescale: rescale outputs using factor in top-level
            SCALE_VALUES mapping if available
    """

    all_scores: Dict[str, float] = {}
    all_profiles: Dict[str, np.ndarray] = {}

    for target, pred, name in zip(targets, predictions, names):

        scores, profiles = score_single_output(target, pred, name, rescale=rescale)
        all_scores.update(scores)
        all_profiles.update(profiles)

    return all_scores, all_profiles

import logging
from typing import Sequence, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

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


def _append_rectified_cloud_if_available(targets, predictions, names):

    qc_key = "cloud_water_mixing_ratio_after_gscond"
    if qc_key in names:
        idx = names.index(qc_key)
        target = targets[idx]
        prediction = predictions[idx]

        rectified = np.where(prediction < 0, 0, prediction)
        targets.append(target)
        predictions.append(rectified)
        names.append(f"{qc_key}_rectified")


def score_multi_output(
    targets: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    names: Sequence[str],
) -> ScoringOutput:

    """
    Score a multi-output model while retaining a flat dictionary
    structure of the output scores.

    Args:
        target: truth values for each model output
        prediction: emulated values for each model output
        names: names of fields being emulated, used for score key
            uniqueness
    """

    all_scores: Dict[str, float] = {}
    all_profiles: Dict[str, np.ndarray] = {}

    targets = list(targets)
    predictions = list(predictions)
    names = list(names)

    _append_rectified_cloud_if_available(targets, predictions, names)

    for target, prediction, name in zip(targets, predictions, names):

        # including for backwards compatibility for precip scores
        # precip has units meters/timestep, scaling assumes timestep is 900s
        if name == "total_precipitation":
            millimeters_in_meter = 1000
            timestep_seconds = 900
            seconds_in_day = 60 * 60 * 24
            precip_scaling = (
                millimeters_in_meter / timestep_seconds * seconds_in_day
            )  # mm/day
            target *= precip_scaling
            prediction *= precip_scaling

        scores, profiles = score(target, prediction)
        # Keys use directory style for wandb chart grouping
        flat_score = {f"{k}/{name}": v for k, v in scores.items()}
        flat_profile = {f"{k}/{name}": v for k, v in profiles.items()}

        all_scores.update(flat_score)
        all_profiles.update(flat_profile)

    return all_scores, all_profiles

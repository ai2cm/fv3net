import logging
from typing import Iterable
from ._wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__file__)

SAMPLE_DIM = "sample"


def _get_regressor(model_type: str, hyperparameters: dict):
    """Reads the model type from the model training config and initializes a regressor
    with hyperparameters from config.

    Args:
        model_type: sklearn model to train
        hyperparameters: should match the hyperparameters accepted by sklearn model

    Returns:
        regressor (varies depending on model type)
    """
    model_type = model_type.replace(" ", "").replace("_", "")
    if "rf" in model_type or "randomforest" in model_type:
        hyperparameters["random_state"] = hyperparameters.get("random_state", 0)
        hyperparameters["n_jobs"] = hyperparameters.get("n_jobs", -1)
        regressor = RandomForestRegressor(**hyperparameters)
    else:
        raise ValueError(
            f"Model type {model_type} not implemented. "
            "Options are "
            " 1) random forest (contains keywords 'rf' "
            "or 'random forest') "
        )
    return regressor


def get_model(
    model_type: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    scaler_type: str,
    scaler_kwargs: dict,
    **hyperparameters,
) -> SklearnWrapper:
    base_regressor = _get_regressor(model_type, hyperparameters)
    batch_regressor = RegressorEnsemble(base_regressor)
    model_wrapper = SklearnWrapper(
        SAMPLE_DIM,
        input_variables,
        output_variables,
        batch_regressor,
        scaler_type=scaler_type,
        scaler_kwargs=scaler_kwargs,
    )
    return model_wrapper

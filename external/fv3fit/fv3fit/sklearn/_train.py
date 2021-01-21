import logging
from .._shared import ModelTrainingConfig

from ._wrapper import SklearnWrapper, RegressorEnsemble
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__file__)

SAMPLE_DIM = "sample"


def _get_regressor(train_config: ModelTrainingConfig):
    """Reads the model type from the model training config and initializes a regressor
    with hyperparameters from config.

    Args:
        train_config: model training configuration

    Returns:
        regressor (varies depending on model type)
    """
    model_type = train_config.model_type.replace(" ", "").replace("_", "")
    if "rf" in model_type or "randomforest" in model_type:
        train_config.hyperparameters["random_state"] = train_config.hyperparameters.get(
            "random_state", 0
        )
        train_config.hyperparameters["n_jobs"] = train_config.hyperparameters.get(
            "n_jobs", -1
        )
        regressor = RandomForestRegressor(**train_config.hyperparameters)
    else:
        raise ValueError(
            f"Model type {train_config.model_type} not implemented. "
            "Options are "
            " 1) random forest (contains keywords 'rf' "
            "or 'random forest') "
        )
    return regressor


def get_model(
    train_config: ModelTrainingConfig,
) -> SklearnWrapper:
    base_regressor = _get_regressor(train_config)
    batch_regressor = RegressorEnsemble(base_regressor)
    model_wrapper = SklearnWrapper(
        SAMPLE_DIM,
        train_config.input_variables,
        train_config.output_variables,
        batch_regressor,
        scaler_type=train_config.scaler_type,
        scaler_kwargs=train_config.scaler_kwargs,
    )
    return model_wrapper

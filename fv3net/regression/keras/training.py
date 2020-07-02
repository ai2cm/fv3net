import logging
from . import models

logger = logging.getLogger(__file__)


def get_model(model_type, input_variables, output_variables, **hyperparameters):
    return getattr(models, model_type)(
        input_variables, output_variables, **hyperparameters
    )

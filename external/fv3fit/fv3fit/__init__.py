from ._shared import Predictor
from ._shared.io import dump, load
from . import keras
from . import sklearn

PRODUCTION_MODEL_TYPES = {"sklearn": ["random_forest"], "keras": ["DenseModel"]}

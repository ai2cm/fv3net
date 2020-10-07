from ._shared import Predictor, ArrayPacker
from . import keras
from . import sklearn

PRODUCTION_MODEL_TYPES = {"sklearn": ["random_forest"], "keras": ["DenseModel"]}

from vcm.cloud import get_fs
from fv3fit import sklearn, keras
import joblib
from ._mapper import SklearnPredictionMapper, KerasPredictionMapper
from typing import Tuple, Union
PredictionMapper = Union[SklearnPredictionMapper, KerasPredictionMapper]


def load_sklearn_model(model_path: str) -> Tuple[sklearn.SklearnWrapper, PredictionMapper]:
    fs_model = get_fs(model_path)
    with fs_model.open(a.model_path, "rb") as f:
        model = joblib.load(f)
    return model, SklearnPredictionMapper

def load_keras_model(model_path: str, model_type: str = 'DenseModel') -> Tuple[keras.Model, PredictionMapper]:
    model_class = keras.get_model_class(model_type)
    model = model_class.load(model_path)
    return model, KerasPredictionMapper

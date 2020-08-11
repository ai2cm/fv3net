from ._mapper import SklearnPredictionMapper, KerasPredictionMapper
from vcm.cloud import get_fs
from fv3fit import sklearn, keras
import joblib
from typing import Tuple, Union
import os

PredictionMapper = Union[SklearnPredictionMapper, KerasPredictionMapper]


def load_sklearn_model(
    model_path: str,
) -> Tuple[sklearn.SklearnWrapper, PredictionMapper]:
    fs_model = get_fs(model_path)
    with fs_model.open(model_path, "rb") as f:
        model = joblib.load(f)
    return model, SklearnPredictionMapper


def load_keras_model(
    model_path: str,
    keras_model_type: str = "DenseModel",
    model_datadir_name: str = "model_data",
) -> Tuple[keras.Model, PredictionMapper]:
    model_class = keras.get_model_class(keras_model_type)
    model = model_class.load(os.path.join(model_path, model_datadir_name))
    return model, KerasPredictionMapper

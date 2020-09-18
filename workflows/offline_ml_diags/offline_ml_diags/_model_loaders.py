from fv3fit.sklearn import SklearnWrapper
from fv3fit.keras import get_model_class, Model
import os


def load_sklearn_model(
    model_path: str, sklearn_pkl_filename: str = "sklearn_model.pkl"
) -> SklearnWrapper:
    return SklearnWrapper.load(os.path.join(model_path, sklearn_pkl_filename))


def load_keras_model(model_path: str, keras_model_type: str = "DenseModel") -> Model:
    model_class = get_model_class(keras_model_type)
    model = model_class.load(model_path)
    return model

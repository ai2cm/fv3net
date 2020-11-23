from fv3fit.sklearn import SklearnWrapper
from fv3fit.keras import get_model_class, Model
import os


def load_sklearn_model(model_path: str,) -> SklearnWrapper:
    return SklearnWrapper.load(model_path)


def load_keras_model(
    model_path: str,
    model_data_dir: str = "model_data",
    keras_model_type: str = "DenseModel",
) -> Model:
    model_class = get_model_class(keras_model_type)
    # type checking thinks this needs > 1 arg
    model = model_class.load(os.path.join(model_path, model_data_dir))   # type: ignore
    return model   # type: ignore

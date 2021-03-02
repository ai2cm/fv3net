import os
import pickle
import yaml
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import compose_left
import tempfile
from typing import Callable, Hashable, Mapping, Sequence
import fsspec

from vcm import safe
from ..._shared import _transforms
from ..._shared.predictor import DATASET_DIM_NAME, Predictor
from loaders._utils import stack_non_vertical
import fv3fit._shared.io


@fv3fit._shared.io.register("all-physics")
class AllPhysicsEmulator(Predictor):

    def __init__(
        self,
        sample_dim_name,
        input_variables,
        output_variables,
        model: tf.keras.Model,
        X_to_arr_func: Callable[[xr.Dataset], np.ndarray],
        y_to_arr_ds_func: Callable[[np.ndarray], Mapping[str, np.ndarray]],
    ):
        """
        Initialize a predictor for the "all physics" emulation
        model trained in a notebook.  This is a quick way to try
        out a prognostic run with it.
        """

        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = model
        self._X_to_arr_func = X_to_arr_func
        self._y_to_arr_ds_func = y_to_arr_ds_func

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        X_for_predict = self._X_to_arr_func(X)
        y = self._model.predict(X_for_predict)
        y_unstacked = self._y_to_arr_ds_func(y)
        y_ds = _y_to_dataset(y_unstacked, X)
        return y_ds

    def predict_columnwise(
        self,
        X: xr.Dataset,
        sample_dims: Sequence[Hashable] = None,
        feature_dim: Hashable = None,
    ) -> xr.Dataset:
        inputs_ = safe.get_variables(X, self.input_variables)
        X_stacked = stack_non_vertical(inputs_)
        output = self.predict(X_stacked).unstack("sample")
        return output

    @classmethod
    def load(cls, path: str) -> object:
        with tempfile.TemporaryDirectory() as dir_:
            fs = fsspec.get_fs_token_paths(path)[0]
            fs.get(path, dir_, recursive=True)

            X_transform = get_notebook_X_transform_func(dir_)
            y_transform = get_notebook_y_transform_func(dir_)
            model = tf.keras.models.load_model(os.path.join(dir_, "model.tf"))
            with open(os.path.join(dir_, "model_options.yaml"), "r") as f:
                model_options = yaml.safe_load(f)

        return cls(
            model=model,
            X_to_arr_func=X_transform,
            y_to_arr_ds_func=y_transform,
            **model_options
        )
        

def get_notebook_X_transform_func(model_path: str):
    """
    Recreates batch -> X transforms from notebook
    """

    X_stacker = _load_stacker(os.path.join(model_path, "X_stacker.yaml"))
    std_info = _load_std_info(os.path.join(model_path, "standardization_info.pkl"))

    pipeline = [
        _transforms.extract_ds_arrays,
        _transforms.standardize(std_info),
        X_stacker.stack,
    ]

    return compose_left(*pipeline)


def get_notebook_y_transform_func(model_path: str):
    """
    Recreates batch -> X transforms from notebook
    """

    y_stacker = _load_stacker(os.path.join(model_path, "y_stacker.yaml"))
    std_info = _load_std_info(os.path.join(model_path, "standardization_info.pkl"))

    pipeline = [
        y_stacker.unstack_orig_featuresize,
        _transforms.unstandardize(std_info),
    ]

    return compose_left(*pipeline)


def _load_stacker(stacker_path: str) -> _transforms.ArrayStacker:
    with open(stacker_path, "r") as f:
        stacker = _transforms.ArrayStacker.load(f)

    return stacker


def _load_std_info(std_info_path: str):
    with open(std_info_path, "rb") as f:
        std_info = pickle.load(f)
    return std_info


def _y_to_dataset(y: Mapping[str, np.ndarray], X: xr.Dataset):

    dataset = {}
    for varname, data in y.items():
        vdims = _infer_dims(data, X.sizes)
        da = xr.DataArray(data=data, dims=vdims)
        dataset[varname] = da

    dataset = xr.Dataset(dataset)
    dataset = dataset.assign_coords(X.coords)

    return dataset


def _infer_dims(arr: np.ndarray, dim_sizes: Mapping[Hashable, int]) -> Sequence[str]:
    arr_dims = []
    for dim_len in arr.shape:
        match_dim = []
        for dim, size in dim_sizes.items():
            if dim_len == size:
                match_dim.append(dim)
        if len(match_dim) > 1:
            raise ValueError(
                "Dim inference ambiguous with multiple same length dimensions. "
                f"{match_dim}"
            )
        else:
            arr_dims.append(match_dim[0])

    return arr_dims

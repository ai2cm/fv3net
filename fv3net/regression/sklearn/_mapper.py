from typing import Mapping

from vcm import safe
import xarray as xr

from loaders import SAMPLE_DIM_NAME
from .wrapper import SklearnWrapper

# TODO: import base mapper class after loaders refactored out of fv3net
DatasetMapper = Mapping[str, xr.Dataset]


class SklearnPredictionMapper(DatasetMapper):
    def __init__(
        self,
        base_mapper: DatasetMapper,
        sklearn_wrapped_model: SklearnWrapper,
        predicted_var_suffix: str = "ml",
        init_time_dim: str = "initial_time",
        z_dim: str = "z",
    ):
        self._base_mapper = base_mapper
        self._model = sklearn_wrapped_model
        self._init_time_dim = init_time_dim
        self._z_dim = z_dim
        self._output_vars_rename = (
            {
                var: var + f"_{predicted_var_suffix.strip('_')}"
                for var in self._model.output_vars_
            }
            if predicted_var_suffix
            else {}
        )

    def _predict(self, ds):
        if set(self._model.input_vars_).issubset(ds.data_vars) is False:
            missing_vars = [
                var
                for var in set(self._model.input_vars_) ^ set(ds.data_vars)
                if var in self._model.input_vars_
            ]
            raise KeyError(
                f"Model feature variables {missing_vars}  not present in dataset."
            )

        ds_ = safe.get_variables(ds, self._model.input_vars_)
        ds_stacked = safe.stack_once(
            ds_,
            SAMPLE_DIM_NAME,
            [dim for dim in ds_.dims if dim != self._z_dim],
            allowed_broadcast_dims=[self._z_dim, self._init_time_dim],
        )
        ds_pred = self._model.predict(ds_stacked, SAMPLE_DIM_NAME).unstack()
        return ds_pred.rename(self._output_vars_rename)

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._base_mapper[key]
        ds_prediction = self._predict(ds)
        return xr.merge([ds, ds_prediction])

    def keys(self):
        return self._base_mapper.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

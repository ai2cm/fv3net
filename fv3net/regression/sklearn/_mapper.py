from typing import Mapping

from vcm import cloud
from vcm import safe
import xarray as xr

from loaders import SAMPLE_DIM_NAME
from .wrapper import SklearnWrapper

# TODO: import base mapper class after loaders refactored out of fv3net
DatasetMapper = Mapping[str, xr.Dataset]


class SklearnPredictionMapper():
    def __init__(
            self,
            base_mapper: DatasetMapper,
            sklearn_wrapped_model: SklearnWrapper,
            init_time_dim: str = "initial_time",
            z_dim: str = "z"
    ):
        self._base_mapper = base_mapper
        self._fs = cloud.get_fs(base_mapper)
        # TODO: This is current an object defined in fv3net.
        # Assumes input/output format as well as how to access
        # feature variable names. Refactor out the wrapped model,
        # or do away with wrapping models for xarray.
        self._model = sklearn_wrapped_model
        self.init_time_dim = init_time_dim
        self.z_dim = z_dim
        
    def _predict(self, ds):
        ds_ = safe.get_variables(ds, self._model.input_vars_)
        ds_stacked = safe.stack_once(
            ds_,
            SAMPLE_DIM_NAME,
            [dim for dim in ds_.dims if dim != self.z_dim],
            allowed_broadcast_dims=[self.z_dim, self.init_time_dim]
        )
        return self._model.predict(ds_stacked, SAMPLE_DIM_NAME).unstack()
    
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

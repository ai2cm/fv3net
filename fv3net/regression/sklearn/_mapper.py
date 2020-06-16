from typing import Mapping

from vcm import safe
import xarray as xr

from loaders import SAMPLE_DIM_NAME
from .wrapper import SklearnWrapper

DATA_SOURCE_DIM = "data_source"
PREDICT_COORD = "predict"
TARGET_COORD = "target"

# TODO: import base mapper class after loaders refactored out of fv3net
DatasetMapper = Mapping[str, xr.Dataset]


class SklearnPredictionMapper(DatasetMapper):
    def __init__(
        self,
        base_mapper: DatasetMapper,
        sklearn_wrapped_model: SklearnWrapper,
        init_time_dim: str = "initial_time",
        z_dim: str = "z",
    ):
        self._base_mapper = base_mapper
        self._model = sklearn_wrapped_model
        self._init_time_dim = init_time_dim
        self._z_dim = z_dim

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
        return self._model.predict(ds_stacked, SAMPLE_DIM_NAME).unstack()
    
    def _insert_prediction(self, ds, ds_pred):
        predicted_vars = ds_pred.data_vars
        nonpredicted_vars = [var for var in ds.data_vars if var not in predicted_vars]
        ds_target = safe.get_variables(
                ds, [var for var in predicted_vars if var in ds.data_vars]) \
            .expand_dims(DATA_SOURCE_DIM) \
            .assign_coords({DATA_SOURCE_DIM: [TARGET_COORD]})
        ds_pred = ds_pred.expand_dims(DATA_SOURCE_DIM) \
            .assign_coords({DATA_SOURCE_DIM: [PREDICT_COORD]})
        return xr.merge([safe.get_variables(ds, nonpredicted_vars), ds_target, ds_pred])

    def _insert_integrated_vars(self):
        utils.insert_column_integrated_vars()

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._base_mapper[key]
        ds_prediction = self._predict(ds)
        return self._insert_prediction(ds, ds_prediction)

    def keys(self):
        return self._base_mapper.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

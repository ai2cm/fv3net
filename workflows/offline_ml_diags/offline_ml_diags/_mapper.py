from typing import Mapping, Union

from vcm import safe, cast_to_datetime, cos_zenith_angle
import xarray as xr

from fv3fit.sklearn import SklearnWrapper
from fv3fit.keras import Model
from loaders.mappers import GeoMapper
from loaders import DERIVATION_DIM

Predictor = Union[SklearnWrapper, Model]

PREDICT_COORD = "predict"
TARGET_COORD = "target"


class PredictionMapper(GeoMapper):
    def __init__(
        self,
        base_mapper: GeoMapper,
        wrapped_model: Predictor,
        z_dim: str = "z",
        rename_vars: Mapping[str, str] = None,
        cos_z_var: str = None,
        grid: xr.Dataset = None,
    ):
        self._base_mapper = base_mapper
        self._model = wrapped_model
        self._z_dim = z_dim
        self._cos_z_var = cos_z_var
        self._grid = grid
        self.rename_vars = rename_vars or {}

    def _predict(self, ds: xr.Dataset) -> xr.Dataset:
        if set(self._model.input_variables).issubset(ds.data_vars) is False:
            missing_vars = [
                var
                for var in set(self._model.input_variables) ^ set(ds.data_vars)
                if var in self._model.input_variables
            ]
            raise KeyError(
                f"Model feature variables {missing_vars}  not present in dataset."
            )

        ds_ = safe.get_variables(ds, self._model.input_variables)
        ds_stacked = safe.stack_once(
            ds_,
            self._model.sample_dim_name,
            [dim for dim in ds_.dims if dim != self._z_dim],
            allowed_broadcast_dims=[self._z_dim],
        )
        ds_stacked = ds_stacked.transpose(self._model.sample_dim_name, self._z_dim)
        ds_pred = self._model.predict(ds_stacked)

        return ds_pred.unstack().rename(self.rename_vars)

    def _insert_cos_zenith_angle(self, time_key: str, ds: xr.Dataset) -> xr.Dataset:
        time = cast_to_datetime(time_key)
        if self._grid is not None:
            cos_z = cos_zenith_angle(time, self._grid["lon"], self._grid["lat"])
            return ds.assign(
                {self._cos_z_var: (self._grid["lon"].dims, cos_z)}  # type: ignore
            )
        else:
            raise ValueError()

    def _insert_prediction(self, ds: xr.Dataset, ds_pred: xr.Dataset) -> xr.Dataset:
        predicted_vars = ds_pred.data_vars
        nonpredicted_vars = [var for var in ds.data_vars if var not in predicted_vars]
        ds_target = (
            safe.get_variables(
                ds, [var for var in predicted_vars if var in ds.data_vars]
            )
            .expand_dims(DERIVATION_DIM)
            .assign_coords({DERIVATION_DIM: [TARGET_COORD]})
        )
        ds_pred = ds_pred.expand_dims(DERIVATION_DIM).assign_coords(
            {DERIVATION_DIM: [PREDICT_COORD]}
        )
        return xr.merge([safe.get_variables(ds, nonpredicted_vars), ds_target, ds_pred])

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._base_mapper[key]
        if self._cos_z_var and self._grid:
            ds = self._insert_cos_zenith_angle(key, ds)
        ds_prediction = self._predict(ds)
        return self._insert_prediction(ds, ds_prediction)

    def keys(self):
        return self._base_mapper.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

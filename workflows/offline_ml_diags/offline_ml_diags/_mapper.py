from typing import Mapping, Union, Sequence

from vcm import safe, DerivedMapping, parse_datetime_from_str
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
        variables: Sequence[str],
        z_dim: str = "z",
        rename_vars: Mapping[str, str] = None,
        grid: xr.Dataset = None,
    ):
        self._base_mapper = base_mapper
        self._model = wrapped_model
        self._z_dim = z_dim
        self._grid = grid
        self._variables = variables
        self.rename_vars = rename_vars or {}

    def _predict(self, ds: xr.Dataset) -> xr.Dataset:
        output = self._model.predict_columnwise(ds, feature_dim=self._z_dim)
        return output.rename(self.rename_vars)

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
        ds = ds.merge(self._grid, compat="override") \
            .assign_coords({"time": parse_datetime_from_str(key)})
        derived_mapping = DerivedMapping(ds)
        ds_derived = derived_mapping.dataset(self._variables)
        ds_prediction = self._predict(ds_derived)
        return self._insert_prediction(ds_derived, ds_prediction)

    def keys(self):
        return self._base_mapper.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

from typing import Mapping, Sequence

from vcm import safe, DerivedMapping, parse_datetime_from_str
import xarray as xr

import fv3fit
from loaders.mappers import GeoMapper
from loaders import DERIVATION_DIM
import warnings

PREDICT_COORD = "predict"
TARGET_COORD = "target"

DELP = "pressure_thickness_of_atmospheric_layer"


class PredictionMapper(GeoMapper):
    def __init__(
        self,
        base_mapper: GeoMapper,
        wrapped_model: fv3fit.Predictor,
        variables: Sequence[str],
        z_dim: str = "z",
        rename_vars: Mapping[str, str] = None,
        grid: xr.Dataset = None,
    ):
        self._base_mapper = base_mapper
        self._model = wrapped_model
        self._z_dim = z_dim
        self._grid = grid or xr.Dataset()
        self._variables = variables
        self.rename_vars = rename_vars or {}

    def _predict(self, ds: xr.Dataset) -> xr.Dataset:
        output = self._model.predict_columnwise(ds, feature_dim=self._z_dim)
        return output.rename(self.rename_vars)  # type: ignore

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
        if "dQ1" not in self._variables and "dQu" in self._variables:
            ds_pred["dQ1"] = xr.zeros_like(ds_pred["dQu"])
            ds_pred["dQ2"] = xr.zeros_like(ds_pred["dQu"])
            ds_target["dQ1"] = xr.zeros_like(ds_target["dQu"])
            ds_target["dQ2"] = xr.zeros_like(ds_target["dQu"])
        return xr.merge([safe.get_variables(ds, nonpredicted_vars), ds_target, ds_pred])

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._base_mapper[key]
        # Prioritize dataset's land_sea_mask if grid values disagree
        ds = xr.merge(
            [ds, self._grid], compat="override"  # type: ignore
        ).assign_coords({"time": parse_datetime_from_str(key)})
        if "dQ1" not in self._variables and "dQu" in self._variables:
            ds["dQ1"] = xr.zeros_like(ds["dQu"])
            ds["dQ2"] = xr.zeros_like(ds["dQu"])
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

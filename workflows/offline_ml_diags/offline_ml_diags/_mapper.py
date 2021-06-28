from typing import Sequence

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
    """A mapper of outputs from a predictor, with inputs coming from a base mapper."""

    def __init__(
        self,
        base_mapper: GeoMapper,
        predictor: fv3fit.Predictor,
        variables: Sequence[str],
        z_dim: str = "z",
        grid: xr.Dataset = None,
    ):
        """
        Args:
            base_mapper: mapper containing input data for prediction
            predictor: model for prediction
            variables: names of variables from base_mapper and grid to include in
                input to predictive model, and in output datasets (alongside any
                predicted outputs)
            z_dim: name of the z-dimension used as a feature dimension
            grid: constant dataset with prediction inputs
        """
        self._base_mapper = base_mapper
        self._model = predictor
        self._z_dim = z_dim
        # TODO: maybe split off responsibility of merging grid with each dataset
        # in base_mapper into its own class, wrapping base_mapper
        self._grid = grid or xr.Dataset()
        # TODO: can this arg be removed, instead using metadata on predictor?
        self._variables = variables

    def _predict(self, ds: xr.Dataset) -> xr.Dataset:
        print(f"drop_levels {self._model.drop_levels}")
        if "z" in ds.dims:
            ds = ds.isel(z=slice(self._model.drop_levels, None))
        result = self._model.predict_columnwise(ds, feature_dim=self._z_dim)
        if "z" in ds.dims:
            result = result.pad(z=(self._model.drop_levels, 0), constant_values=0.0)
        return result


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
        # Prioritize dataset's land_sea_mask if grid values disagree
        ds = xr.merge(
            [ds, self._grid], compat="override"  # type: ignore
        ).assign_coords({"time": parse_datetime_from_str(key)})
        derived_mapping = DerivedMapping(ds)

        ds_derived = xr.Dataset({})
        for key in self._variables:
            try:
                ds_derived[key] = derived_mapping[key]
            except KeyError as e:
                if key == DELP:
                    raise e
                elif key in ["pQ1", "pQ2", "dQ1", "dQ2"]:
                    ds_derived[key] = xr.zeros_like(derived_mapping[DELP])
                    warnings.warn(
                        f"{key} not present in data. Filling with zeros.", UserWarning
                    )
                else:
                    raise e

        ds_prediction = self._predict(ds_derived)
        return self._insert_prediction(ds_derived, ds_prediction)

    def keys(self):
        return self._base_mapper.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

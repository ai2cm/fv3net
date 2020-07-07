from typing import Sequence
import xarray as xr
from diagnostics_utils import insert_column_integrated_vars

from fv3net.regression.sklearn import TARGET_COORD, PREDICT_COORD, DERIVATION_DIM


DELP_VAR = "pressure_thickness_of_atmospheric_layer"
AREA_VAR = "area"

# Variables predicted by model
COL_INTEGRATE_VARS = ["dQ1", "dQ2", "pQ1", "pQ2", "Q1", "Q2"]
# Variables to calculate RMSE and bias of
METRIC_VARS = ["dQ1", "dQ2", "column_integrated_dQ1", "column_integrated_dQ2"]


def insert_additional_variables(ds, area):
    ds["area_weights"] = area / (area.mean())
    ds["delp_weights"] = ds[DELP_VAR] / ds[DELP_VAR].mean("z")
    ds["Q1"] = ds["pQ1"] + ds["dQ1"]
    ds["Q2"] = ds["pQ2"] + ds["dQ2"]
    ds = insert_column_integrated_vars(ds, COL_INTEGRATE_VARS)
    ds = _insert_means(ds, METRIC_VARS, ds["area_weights"])
    return ds


def _insert_means(
    ds: xr.Dataset, vars: Sequence[str], weights: xr.DataArray = None
) -> xr.Dataset:
    for var in vars:
        da = ds[var].sel({DERIVATION_DIM: [TARGET_COORD, PREDICT_COORD]})
        weights = 1.0 if weights is None else weights
        mean = (
            (da.sel({DERIVATION_DIM: TARGET_COORD}) * weights)
            .mean()
            .assign_coords({DERIVATION_DIM: "mean"})
        )
        da = xr.concat([da, mean], dim=DERIVATION_DIM)
        ds = ds.drop([var])
        ds = ds.merge(da)
    return ds

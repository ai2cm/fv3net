import json
import numpy as np
import os
import shutil
from typing import Mapping, Sequence, Dict, Tuple
import vcm
import xarray as xr

from vcm import safe
from vcm.cloud import gsutil
from vcm.catalog import catalog

DELP = "pressure_thickness_of_atmospheric_layer"


UNITS = {
    "column_integrated_dq1": "[W/m2]",
    "column_integrated_dq2": "[mm/day]",
    "column_integrated_q1": "[W/m2]",
    "column_integrated_q2": "[mm/day]",
    "column_integrated_dqu": "[Pa]",
    "column_integrated_dqv": "[Pa]",
    "dq1": "[K/s]",
    "pq1": "[K/s]",
    "q1": "[K/s]",
    "dq2": "[kg/kg/s]",
    "pq2": "[kg/kg/s]",
    "q2": "[kg/kg/s]",
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface": "[W/m2]",
    "net_shortwave_sfc_flux_derived": "[W/m2]",
    "total_precipitation_rate": "[kg/m2/s]",
    "water_vapor_path": "[mm]",
    "minus_column_integrated_q2": "[mm/day]",
}
UNITS = {**UNITS, **{f"error_in_{k}": v for k, v in UNITS.items()}}
UNITS = {**UNITS, **{f"{k}_snapshot": v for k, v in UNITS.items()}}

GRID_INFO_VARS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
    "lat",
    "lon",
    "latb",
    "lonb",
    "land_sea_mask",
    "area",
]
ScalarMetrics = Dict[str, Mapping[str, float]]


def is_3d(da: xr.DataArray, vertical_dim: str = "z"):
    return vertical_dim in da.dims


def insert_r2(ds_metrics):
    mse_vars = [var for var in ds_metrics if "_mse" in var]
    for mse_var in mse_vars:
        variance = ds_metrics[mse_var.replace("_mse", "_variance")]
        ds_metrics[mse_var.replace("_mse", "_r2")] = (
            1.0 - ds_metrics[mse_var] / variance
        )
    return ds_metrics


def insert_rmse(ds: xr.Dataset):
    mse_vars = [var for var in ds.data_vars if "_mse" in str(var)]
    for mse_var in mse_vars:
        rmse_var = str(mse_var).replace("_mse", "_rmse")
        ds[rmse_var] = np.sqrt(ds[mse_var])
    return ds


def load_grid_info(res: str = "c48"):
    grid = catalog[f"grid/{res}"].read()
    wind_rotation = catalog[f"wind_rotation/{res}"].read()
    land_sea_mask = catalog[f"landseamask/{res}"].read()
    grid_info = xr.merge([grid, wind_rotation, land_sea_mask])
    return safe.get_variables(grid_info, GRID_INFO_VARS).drop("tile")


def open_diagnostics_outputs(
    data_dir,
    diagnostics_nc_name: str,
    transect_nc_name: str,
    metrics_json_name: str,
    metadata_json_name: str,
) -> Tuple[xr.Dataset, xr.Dataset, dict, dict]:
    fs = vcm.get_fs(data_dir)
    with fs.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    transect_full_path = os.path.join(data_dir, transect_nc_name)
    if fs.exists(transect_full_path):
        with fs.open(transect_full_path, "rb") as f:
            ds_transect = xr.open_dataset(f).load()
    else:
        ds_transect = xr.Dataset()
    with fs.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    with fs.open(os.path.join(data_dir, metadata_json_name), "r") as f:
        metadata = json.load(f)
    return ds_diags, ds_transect, metrics, metadata


def copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def tidy_title(var: str):
    title = (
        var.replace("pressure_level", "")
        .replace("zonal_avg_pressure_level", "")
        .replace("-", " ")
    )
    return title[0].upper() + title[1:]


def get_metric_string(
    metric_statistics: Mapping[str, float], precision=2,
):
    value = metric_statistics["mean"]
    std = metric_statistics["std"]
    return " ".join(["{:.2e}".format(value), "+/-", "{:.2e}".format(std)])


def units_from_name(var):
    for key, units in UNITS.items():
        # allow additional suffixes on variable
        if key in var.lower():
            return units
        else:
            return "[units unavailable]"


def insert_column_integrated_vars(
    ds: xr.Dataset, column_integrated_vars: Sequence[str]
) -> xr.Dataset:
    """Insert column integrated (<*>) terms,
    really a wrapper around vcm.calc.thermo funcs"""

    for var in column_integrated_vars:
        column_integrated_name = f"column_integrated_{var}"
        if "Q1" in var:
            da = vcm.column_integrated_heating_from_isochoric_transition(
                ds[var], ds[DELP]
            )
        elif "Q2" in var:
            da = -vcm.minus_column_integrated_moistening(ds[var], ds[DELP])
            da = da.assign_attrs(
                {"long_name": "column integrated moistening", "units": "mm/day"}
            )
        else:
            da = vcm.mass_integrate(ds[var], ds[DELP], dim="z")
        ds = ds.assign({column_integrated_name: da})

    return ds

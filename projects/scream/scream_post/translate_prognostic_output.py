import argparse
import xarray as xr
from typing import Mapping, Hashable, List
import vcm
import os
from dask.distributed import Client
from util import (
    convert_npdatetime_to_cftime,
    rename_lev_to_z,
    split_horiz_winds_tend,
    add_rad_fluxes,
)
import re
import logging

logger = logging.getLogger("translate scream run")

SCREAM_TO_FV3_CONVENTION = {
    "T_mid": "air_temperature",
    "qv": "specific_humidity",
    "U": "eastward_wind",
    "V": "northward_wind",
    "nudging_T_mid_tend": "air_temperature_tendency_due_to_nudging",
    "nudging_qv_tend": "specific_humidity_tendency_due_to_nudging",
    "nudging_U_tend": "eastward_wind_tendency_due_to_nudging",
    "nudging_V_tend": "northward_wind_tendency_due_to_nudging",
    "T_mid_tendency_due_to_nudging": "air_temperature_tendency_due_to_nudging",
    "U_tendency_due_to_nudging": "eastward_wind_tendency_due_to_nudging",
    "V_tendency_due_to_nudging": "northward_wind_tendency_due_to_nudging",
    "qv_tendency_due_to_nudging": "specific_humidity_tendency_due_to_nudging",
    "physics_T_mid_tend": "air_temperature_tendency_due_to_scream_physics",
    "physics_qv_tend": "specific_humidity_tendency_due_to_scream_physics",
    "physics_U_tend": "eastward_wind_tendency_due_to_scream_physics",
    "physics_V_tend": "northward_wind_tendency_due_to_scream_physics",
    "machine_learning_U_tend": "dQu",
    "machine_learning_V_tend": "dQv",
    "machine_learning_T_mid_tend": "dQ1",
    "machine_learning_qv_tend": "dQ2",
    "VapWaterPath": "water_vapor_path",
    "ps": "PRESsfc",
    "pseudo_density": "pressure_thickness_of_atmospheric_layer",
    "surf_evap": "surf_evap",
    "surf_sens_flux": "SHTFLsfc",
    "surf_radiative_T": "TMPsfc",
    "z_mid_at_500hPa": "h500",
    "T_mid_at_850hPa": "TMP850",
    "T_mid_at_700hPa": "TMP700",
    "T_mid_at_500hPa": "TMP500",
    "T_mid_at_200hPa": "TMP200",
    "omega_at_500hPa": "w500",
    "RelativeHumidity_at_850hPa": "RH850",
    "RelativeHumidity_at_700hPa": "RH700",
    "RelativeHumidity_at_500hPa": "RH500",
    "SeaLevelPressure": "PRMSL",
    "SW_flux_dn_at_model_bot": "DSWRFsfc",
    "LW_flux_dn_at_model_bot": "DLWRFsfc",
    "LW_flux_up_at_model_bot": "ULWRFsfc",
    "SW_flux_up_at_model_bot": "USWRFsfc",
    "SW_flux_up_at_model_top": "USWRFtoa",
    "LW_flux_up_at_model_top": "ULWRFtoa",
    "SW_flux_dn_at_model_top": "DSWRFtoa",
    "lat": "lat",
    "lon": "lon",
    "area": "area",
    "PRATEsfc": "PRATEsfc",
    "shortwave_transmissivity_of_atmospheric_column": "shortwave_transmissivity_of_atmospheric_column",  # noqa: E501
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",  # noqa: E501
}

RENAME_ML_CORRECTION_TO_MACHINE_LEARNING = {
    "mlcorrection_T_mid_tend": "machine_learning_T_mid_tend",
    "mlcorrection_qv_tend": "machine_learning_qv_tend",
    "mlcorrection_horiz_winds_tend": "machine_learning_horiz_winds_tend",
}

TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
EASTWARD_WIND = "eastward_wind"
NORTHWARD_WIND = "northward_wind"


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_data",
        type=str,
        default=None,
        help="Input netcdf file(s) in string, wildcards allowed.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where output will be written.",
    )
    parser.add_argument(
        "chunk_size",
        type=int,
        help="Chunk size for the time dimension of output zarrs.",
    )
    parser.add_argument("variable_category", type=str, help="Either 2d, 3d, or all.")
    parser.add_argument(
        "--subset",
        help="Whether to subset to a smaller set of timesteps",
        action="store_true",
    )
    return parser


def compute_PRATEsfc(ds):
    """
    Compute PRATEsfc from precip_liq_surf_mass_flux and precip_ice_surf_mass_flux
    """
    if "PrecipIceSurfMassFlux" in ds.variables:
        ds = ds.rename({"PrecipIceSurfMassFlux": "precip_liq_surf_mass_flux"})
    if "PrecipLiqSurfMassFlux" in ds.variables:
        ds = ds.rename({"PrecipLiqSurfMassFlux": "precip_ice_surf_mass_flux"})
    # scream native output saves "mass flux" as m s^-1
    # we multiply by the water density (1000) to get to mass flux (kg m^-2 s^-1)
    PRATEsfc = (ds.precip_liq_surf_mass_flux + ds.precip_ice_surf_mass_flux) * 1000.0
    PRATEsfc = PRATEsfc.assign_attrs(units="kg/m^2/s")
    ds["PRATEsfc"] = PRATEsfc
    return ds


def add_additional_surface_fields(ds):
    """
    Add shortwave_transmissivity_of_atmospheric_column
    and override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface
    for training ML radiaitve fluxes
    Add PRATEsfc for training diagnostic purposes
    """
    ds = add_rad_fluxes(ds)
    ds = compute_PRATEsfc(ds)
    return ds


def determine_nudging_or_ML_run(ds):
    label = None
    for var in ds.variables:
        if "nudging" in var:
            label = "nudging"
        elif "machine_learning" in var:
            label = "machine_learning"
    return label


def compute_prognostic_run_diagnostics(
    ds: xr.DataArray, label: str, tendency_variables: List, hydrostatic: bool = False
):
    """
    Compute diagnostics for a prognostic run

    Args:
        ds: xarray dataset containing prognostic run output
        label: either "machine_learning" or "nudging"
        tendency_variables: list of variables that are either nudged or ML corrected
        hydrostatic: whether the run is hydrostatic
    """
    delp = ds[DELP]
    temperature_tendency_name = "dQ1"
    humidity_tendency_name = "dQ2"
    if temperature_tendency_name and humidity_tendency_name in ds.variables:
        temperature_tendency = ds[temperature_tendency_name]
        humidity_tendency = ds[humidity_tendency_name]
    elif label == "nudging":
        # In fv3net prognostic run,
        # the nudging tendencies are also called "dQ1" and "dQ2"
        temperature_tendency = ds["air_temperature_tendency_due_to_nudging"]
        humidity_tendency = ds["specific_humidity_tendency_due_to_nudging"]
    else:
        temperature_tendency = xr.zeros_like(delp)
        humidity_tendency = xr.zeros_like(delp)

    # compute column-integrated diagnostics
    if hydrostatic:
        net_heating = vcm.column_integrated_heating_from_isobaric_transition(
            temperature_tendency, delp, "z"
        )
    else:
        net_heating = vcm.column_integrated_heating_from_isochoric_transition(
            temperature_tendency, delp, "z"
        )
    diags = {
        f"net_moistening_due_to_{label}": vcm.mass_integrate(
            humidity_tendency, delp, dim="z"
        ).assign_attrs(
            units="kg/m^2/s",
            description=f"column integrated moisture tendency due to {label}",
        ),
        f"column_heating_due_to_{label}": net_heating.assign_attrs(
            units="W/m^2"
        ).assign_attrs(description=f"column integrated heating due to {label}"),
    }
    # add 3D tendencies to diagnostics
    if label == "nudging":
        diags_3d: Mapping[Hashable, xr.DataArray] = {
            SCREAM_TO_FV3_CONVENTION[f"nudging_{k}_tend"]: ds[
                SCREAM_TO_FV3_CONVENTION[f"nudging_{k}_tend"]
            ]
            for k in tendency_variables
        }
    elif label == "machine_learning":
        diags_3d = {
            "dQ1": temperature_tendency.assign_attrs(units="K/s").assign_attrs(
                description=f"air temperature tendency due to {label}"
            ),
            "dQ2": humidity_tendency.assign_attrs(units="kg/kg/s").assign_attrs(
                description=f"specific humidity tendency due to {label}"
            ),
        }
    diags.update(diags_3d)
    if (
        label == "machine_learning"
        and "U" in tendency_variables
        and "V" in tendency_variables
    ):
        diags.update(compute_ml_momentum_diagnostics(ds))
    return xr.Dataset(diags)


def compute_ml_momentum_diagnostics(ds: xr.DataArray):
    delp = ds[DELP]
    dQu = ds["dQu"]
    dQv = ds["dQv"]
    column_integrated_dQu = vcm.mass_integrate(dQu, delp, "z")
    column_integrated_dQv = vcm.mass_integrate(dQv, delp, "z")
    momentum = dict(
        dQu=dQu.assign_attrs(units="m s^-2").assign_attrs(
            description="zonal wind tendency due to machine_learning"
        ),
        dQv=dQv.assign_attrs(units="m s^-2").assign_attrs(
            description="meridional wind tendency due to machine_learning"
        ),
        column_integrated_dQu_stress=column_integrated_dQu.assign_attrs(
            units="Pa",
            description="column integrated zonal wind tendency due to machine_learning",
        ),
        column_integrated_dQv_stress=column_integrated_dQv.assign_attrs(
            units="Pa",
            description="column integrated meridional wind tendency due to machine_learning",  # noqa: E501
        ),
    )
    return xr.Dataset(momentum)


def rename_var_from_scream_to_fv3(ds):
    rename_vars = {k: v for k, v in SCREAM_TO_FV3_CONVENTION.items() if k in ds}
    ds = ds.rename(rename_vars)
    return ds


def get_run_label_and_tendency_var_and_rename(ds: xr.Dataset):
    ds = rename_ML_correction_to_machine_learning(ds)
    run_label = determine_nudging_or_ML_run(ds)
    tendency_variables = determine_tendency_variables(ds, run_label)
    if f"{run_label}_horiz_winds_tend" in ds.variables:
        ds = split_horiz_winds_tend(ds, run_label)
        tendency_variables.remove("horiz_winds")
        tendency_variables += ["U", "V"]
    if "physics_horiz_winds_tend" in ds.variables:
        ds = split_horiz_winds_tend(ds, "physics")
    ds = rename_var_from_scream_to_fv3(ds)
    return run_label, tendency_variables, ds


def convert_time_and_change_lev_to_z(ds):
    ds = convert_npdatetime_to_cftime(ds)
    ds = rename_lev_to_z(ds)
    return ds


def get_3d_vars(ds: xr.Dataset):
    grid_var = ["lat", "lon", "area"]
    var_3d = [
        var for var in ds.variables if set(ds[var].dims) == set(("time", "ncol", "z"))
    ]
    return grid_var + var_3d


def get_2d_vars(ds: xr.Dataset):
    grid_var = ["lat", "lon", "area"]
    var_2d = [var for var in ds.variables if set(ds[var].dims) == set(("time", "ncol"))]
    return grid_var + var_2d


def rename_ML_correction_to_machine_learning(ds: xr.Dataset):
    logger.info("Renaming ML correction to machine learning")
    rename_vars = {
        k: v for k, v in RENAME_ML_CORRECTION_TO_MACHINE_LEARNING.items() if k in ds
    }
    return ds.rename(rename_vars)


def _get_variable_names(ds: xr.Dataset, pattern: str):

    variable_names = []
    for var in ds.variables:
        match = re.match(pattern, var)
        if match:
            variable_names.append(match.group(1)) 
    
    return variable_names

def determine_tendency_variables(ds: xr.Dataset, run_label: str):
    # from original scream output files
    tendency_variables = _get_variable_names(ds, rf"^{run_label}_(.*)_tend$")
    # from zarr files produced by format_output.py
    tendency_variables += _get_variable_names(ds, rf"^(.*?)_tendency_due_to_nudging")
    return list(set(tendency_variables))


def open_dataset(input_data: str):
    if "*" in input_data and ".nc" in input_data:
        logger.info("Opening multiple netcdf files")
        ds = xr.open_mfdataset(input_data, compat="override", coords="minimal")
    elif ".nc" in input_data:
        logger.info("Opening a single netcdf file")
        ds = xr.open_dataset(input_data, use_cftime=True)
    else:
        dataset_members = (
            "physics_tendencies.zarr",
            "nudging_tendencies.zarr",
            "state_after_timestep.zarr",
        )
        logger.info("Opening directory of zarrs produced by scream-post format_output.py")
        ds = xr.merge(
            [xr.open_zarr(os.path.join(input_data, member), use_cftime=True) for member in dataset_members]
        )
    return ds


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    client = Client(n_workers=50, threads_per_worker=2)
    ds = open_dataset(args.input_data)
    if args.subset:
        logger.info("Subset to the first 100 timesteps")
        ds = ds.isel(time=slice(100))
    run_label, tendency_variables, ds = get_run_label_and_tendency_var_and_rename(ds)
    ds = convert_time_and_change_lev_to_z(ds)
    if run_label:
        diags = compute_prognostic_run_diagnostics(ds, run_label, tendency_variables)
        ds = ds.drop(
            [var for var in ds.variables if var in diags.variables and var != "time"]
        )
        ds = xr.merge([ds, diags])
    ds = add_additional_surface_fields(ds)
    ds = ds.chunk({"time": args.chunk_size})
    ds_2d = ds[get_2d_vars(ds)]
    ds_3d = ds[get_3d_vars(ds)]
    if args.variable_category == "2d":
        ds_2d.to_zarr(os.path.join(args.output_path, "data_2d.zarr"), safe_chunks=False)
    elif args.variable_category == "3d":
        ds_3d.to_zarr(os.path.join(args.output_path, "data_3d.zarr"), safe_chunks=False)
    elif args.variable_category == "all":
        ds_2d.to_zarr(os.path.join(args.output_path, "data_2d.zarr"))
        ds_3d.to_zarr(os.path.join(args.output_path, "data_3d.zarr"))

import argparse
import xarray as xr
from typing import Mapping, Hashable
import vcm
import os
from util import convert_npdatetime_to_cftime, rename_lev_to_z, split_horiz_winds_tend

SCREAM_TO_FV3_CONVENTION = {
    "T_mid": "air_temperature",
    "qv": "specific_humidity",
    "U": "eastward_wind",
    "V": "northward_wind",
    "nudging_T_mid_tend": "air_temperature_tendency_due_to_nudging",
    "nudging_qv_tend": "specific_humidity_tendency_due_to_nudging",
    "nudging_U_tend": "eastward_wind_tendency_due_to_nudging",
    "nudging_V_tend": "northward_wind_tendency_due_to_nudging",
    "physics_T_mid_tend": "air_temperature_tendency_due_to_scream_physics",
    "physics_qv_tend": "specific_humidity_tendency_due_to_scream_physics",
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
}

RENAME_ML_CORRECTION_TO_MACHINE_LEARNING = {
    "mlcorrection_T_mid_tend": "machine_learning_T_mid_tend",
    "mlcorrection_qv_tend": "machine_learning_qv_tend",
    "mlcorrection_horiz_winds_tend": "machine_learning_horiz_winds_tend",
}

TEMP = "T_mid"
SPHUM = "qv"
DELP = "pseudo_density"
EASTWARD_WIND = "U"
NORTHWARD_WIND = "V"


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_data", type=str, default=None, help=("Input netcdf."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where output will be written.",
    )
    parser.add_argument(
        "process_label", type=str, help="Either nudging or machine_learning",
    )
    parser.add_argument(
        "nudging_variables",
        type=str,
        help="List of nudging variables deliminated with commas",
    )
    parser.add_argument("chunk_size", type=int, help="Chunk size for output zarrs.")
    parser.add_argument("save_to_disk", type=str, help="Either 2d or 3d.")
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
    PRATEsfc = (ds.precip_liq_surf_mass_flux + ds.precip_ice_surf_mass_flux) * 1000.0
    PRATEsfc = PRATEsfc.assign_attrs(units="kg/m^2/s")
    return PRATEsfc


def add_additional_surface_fields(ds):
    """
    Add shortwave_transmissivity_of_atmospheric_column
    and override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface
    for training ML radiaitve fluxes
    """
    shortwave_transmissivity_of_atmospheric_column = (
        ds.SW_flux_dn_at_model_bot / ds.SW_flux_dn_at_model_top
    )
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.where(  # noqa: E501
        ds.SW_flux_dn_at_model_top != 0.0, 0.0
    )
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.assign_attrs(  # noqa: E501
        units="-", long_name="shortwave transmissivity of atmospheric column"
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = (
        ds.LW_flux_dn_at_model_bot
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface.assign_attrs(  # noqa: E501
        units="W/m**2", long_name="surface downward longwave flux"
    )
    ds[
        "shortwave_transmissivity_of_atmospheric_column"
    ] = shortwave_transmissivity_of_atmospheric_column
    ds[
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
    ] = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface
    return ds


def compute_prognostic_run_diagnostics(
    ds: xr.DataArray, label: str, tendency_variables: [], hydrostatic: bool = False
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
    temperature_tendency_name = f"{label}_{TEMP}_tend"
    humidity_tendency_name = f"{label}_{SPHUM}_tend"
    temperature_tendency = ds[temperature_tendency_name]
    humidity_tendency = ds[humidity_tendency_name]
    water_vapor_path = ds["VapWaterPath"]
    # compute column-integrated diagnostics
    if hydrostatic:
        net_heating = vcm.column_integrated_heating_from_isobaric_transition(
            temperature_tendency, delp, "lev"
        )
    else:
        net_heating = vcm.column_integrated_heating_from_isochoric_transition(
            temperature_tendency, delp, "lev"
        )
    diags = {
        f"net_moistening_due_to_{label}": vcm.mass_integrate(
            humidity_tendency, delp, dim="lev"
        ).assign_attrs(
            units="kg/m^2/s",
            description=f"column integrated moisture tendency due to {label}",
        ),
        f"column_heating_due_to_{label}": net_heating.assign_attrs(
            units="W/m^2"
        ).assign_attrs(description=f"column integrated heating due to {label}"),
        f"water_vapor_path": water_vapor_path.assign_attrs(units="kg/m^2").assign_attrs(
            description=f"water vapor path"
        ),
    }
    # add 3D tendencies to diagnostics
    if label == "nudging":
        diags_3d: Mapping[Hashable, xr.DataArray] = {
            f"{k}_tendency_due_to_nudging": ds[f"{label}_{k}_tend"]
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
    dQu = ds["machine_learning_U_tend"]
    dQv = ds["machine_learning_V_tend"]
    column_integrated_dQu = vcm.mass_integrate(dQu, delp, "lev")
    column_integrated_dQv = vcm.mass_integrate(dQv, delp, "lev")
    momentum = dict(
        dQu=dQu.assign_attrs(units="m s^-2").assign_attrs(
            description="zonal wind tendency due to ML"
        ),
        dQv=dQv.assign_attrs(units="m s^-2").assign_attrs(
            description="meridional wind tendency due to ML"
        ),
        column_integrated_dQu_stress=column_integrated_dQu.assign_attrs(
            units="Pa", description="column integrated zonal wind tendency due to ML",
        ),
        column_integrated_dQv_stress=column_integrated_dQv.assign_attrs(
            units="Pa",
            description="column integrated meridional wind tendency due to ML",
        ),
    )
    return xr.Dataset(momentum)


def truncate_and_rename_scream_variables(ds: xr.Dataset):
    output_var_list = [
        var for var in SCREAM_TO_FV3_CONVENTION.keys() if var in ds.variables
    ]
    grid_var = ["lat", "lon", "area"]
    output_var_list += grid_var
    ds["PRATEsfc"] = compute_PRATEsfc(ds)
    ds = add_additional_surface_fields(ds)
    output_var_list += [
        "PRATEsfc",
        "shortwave_transmissivity_of_atmospheric_column",
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
    ]
    ds = ds[output_var_list]
    rename_vars = {k: v for k, v in SCREAM_TO_FV3_CONVENTION.items() if k in ds}
    ds = ds.rename(rename_vars)
    return ds


def convert_time_and_change_lev_to_z(ds):
    ds = convert_npdatetime_to_cftime(ds)
    ds = rename_lev_to_z(ds)
    return ds


def get_3d_vars(ds: xr.Dataset):
    return [var for var in ds.variables if ds[var].dims == ("time", "ncol", "z")]


def get_2d_vars(ds: xr.Dataset):
    return [var for var in ds.variables if ds[var].dims == ("time", "ncol")]


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    ds = xr.open_mfdataset(args.input_data)
    if args.subset:
        ds = ds.isel(time=slice(100))
    nudging_vars = [str(item) for item in args.nudging_variables.split(",")]
    if args.process_label == "machine_learning":
        rename_vars = {
            k: v for k, v in RENAME_ML_CORRECTION_TO_MACHINE_LEARNING.items() if k in ds
        }
        ds = ds.rename(rename_vars)
    if "U" in nudging_vars or "V" in nudging_vars:
        ds = split_horiz_winds_tend(ds, args.process_label)
    if args.process_label == "nudging" or args.process_label == "machine_learning":
        diags = compute_prognostic_run_diagnostics(ds, args.process_label, nudging_vars)
        ds = truncate_and_rename_scream_variables(ds)
        ds = ds.drop(
            [var for var in ds.variables if var in diags.variables and var != "time"]
        )
        ds = xr.merge([ds, diags])
    else:
        ds = truncate_and_rename_scream_variables(ds)
    ds = convert_time_and_change_lev_to_z(ds)
    grid_var = ["lat", "lon", "area"]
    ds_2d = ds[grid_var + get_2d_vars(ds)]
    ds_3d = ds[grid_var + get_3d_vars(ds)]
    if args.save_to_disk == "2d":
        ds_2d.chunk({"time": args.chunk_size}).to_zarr(
            os.path.join(args.output_path, "data_2d.zarr"), consolidated=True
        )
    elif args.save_to_disk == "3d":
        ds_3d.chunk({"time": args.chunk_size}).to_zarr(
            os.path.join(args.output_path, "data_3d.zarr"), consolidated=True
        )
    elif args.save_to_disk == "all":
        ds_2d.chunk({"time": args.chunk_size}).to_zarr(
            os.path.join(args.output_path, "data_2d.zarr"), consolidated=True
        )
        ds_3d.chunk({"time": args.chunk_size}).to_zarr(
            os.path.join(args.output_path, "data_3d.zarr"), consolidated=True
        )

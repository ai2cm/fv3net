import xarray
from typing import Tuple
from typing_extensions import Protocol
import vcm


def eddy_flux_coarse(unresolved_flux, total_resolved_flux, omega, field):
    """Compute re-coarsened eddy flux divergence from re-coarsed data
    """
    return unresolved_flux + (total_resolved_flux - omega * field)


FINE_RES_BUDGET_NAMES = {
    "T": "air_temperature",
    "sphum": "specific_humidity",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "DLWRFsfc_coarse": "total_sky_downward_longwave_flux_at_surface",
    "DSWRFsfc_coarse": "total_sky_downward_shortwave_flux_at_surface",
    "DSWRFtoa_coarse": "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "ULWRFsfc_coarse": "total_sky_upward_longwave_flux_at_surface",
    "ULWRFtoa_coarse": "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "USWRFsfc_coarse": "total_sky_upward_shortwave_flux_at_surface",
    "USWRFtoa_coarse": "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "LHTFLsfc_coarse": "latent_heat_flux",
    "SHTFLsfc_coarse": "sensible_heat_flux",
    "PRATEsfc_coarse": "surface_precipitation_rate",
}


class FineResBudget(Protocol):
    """Protocol defining what input vaiables are required

    Only used for type checking and editor autocompletion.
    """

    area: xarray.DataArray
    delp: xarray.DataArray
    T: xarray.DataArray
    dq3dt_deep_conv_coarse: xarray.DataArray
    dq3dt_mp_coarse: xarray.DataArray
    dq3dt_pbl_coarse: xarray.DataArray
    dq3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_deep_conv_coarse: xarray.DataArray
    dt3dt_lw_coarse: xarray.DataArray
    dt3dt_mp_coarse: xarray.DataArray
    dt3dt_ogwd_coarse: xarray.DataArray
    dt3dt_pbl_coarse: xarray.DataArray
    dt3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_sw_coarse: xarray.DataArray
    eddy_flux_vulcan_omega_sphum: xarray.DataArray
    eddy_flux_vulcan_omega_temp: xarray.DataArray
    exposed_area: xarray.DataArray
    qv_dt_fv_sat_adj_coarse: xarray.DataArray
    qv_dt_phys_coarse: xarray.DataArray
    sphum: xarray.DataArray
    sphum_storage: xarray.DataArray
    sphum_vulcan_omega_coarse: xarray.DataArray
    t_dt_fv_sat_adj_coarse: xarray.DataArray
    t_dt_nudge_coarse: xarray.DataArray
    t_dt_phys_coarse: xarray.DataArray
    vulcan_omega_coarse: xarray.DataArray
    T_vulcan_omega_coarse: xarray.DataArray
    T_storage: xarray.DataArray
    DLWRFsfc_coarse: xarray.DataArray
    DSWRFsfc_coarse: xarray.DataArray
    DSWRFtoa_coarse: xarray.DataArray
    ULWRFsfc_coarse: xarray.DataArray
    ULWRFtoa_coarse: xarray.DataArray
    USWRFsfc_coarse: xarray.DataArray
    USWRFtoa_coarse: xarray.DataArray
    LHTFLsfc_coarse: xarray.DataArray
    SHTFLsfc_coarse: xarray.DataArray
    PRATEsfc_coarse: xarray.DataArray


def apparent_heating(data: FineResBudget, include_temperature_nudging: bool = False):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_temp,
        data.T_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        data.T,
    )
    eddy_flux_convergence = vcm.convergence_cell_center(eddy_flux, data.delp, dim="z")
    result = data.t_dt_fv_sat_adj_coarse + data.t_dt_phys_coarse + eddy_flux_convergence
    description = (
        "Apparent heating due to physics and sub-grid-scale advection. Given "
        "by sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
    )
    if include_temperature_nudging:
        result = result + data.t_dt_nudge_coarse
        description = description + " + temperature nudging"
    return result.assign_attrs(
        units="K/s",
        long_name="apparent heating from high resolution data",
        description=description,
    ).rename("Q1")


def apparent_moistening(data: FineResBudget):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_sphum,
        data.sphum_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        data.sphum,
    )
    eddy_flux_convergence = vcm.convergence_cell_center(eddy_flux, data.delp, dim="z")
    return (
        (data.qv_dt_fv_sat_adj_coarse + data.qv_dt_phys_coarse + eddy_flux_convergence)
        .assign_attrs(
            units="kg/kg/s",
            long_name="apparent moistening from high resolution data",
            description=(
                "Apparent moistening due to physics and sub-grid-scale advection. "
                "Given by "
                "sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
            ),
        )
        .rename("Q2")
    )


def column_integrated_fine_res_nudging_heating(data: FineResBudget) -> xarray.DataArray:
    heating_in_energy_units = vcm.internal_energy(data.t_dt_nudge_coarse)
    column_heating = vcm.mass_integrate(heating_in_energy_units, data.delp, dim="z")
    return column_heating.assign_attrs(
        units="W/m**2",
        long_name="Column integrated heating tendency due to temperature "
        "nudging of fine-res run.",
    )


def compute_fine_res_sources(
    data: FineResBudget, include_temperature_nudging: bool = False
) -> Tuple[xarray.DataArray, xarray.DataArray]:
    heating = apparent_heating(data, include_temperature_nudging)
    moistening = apparent_moistening(data)
    return heating, moistening

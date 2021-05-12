import xarray as xr
import logging
from runtime.types import State, Diagnostics
from runtime.names import TEMP, SPHUM, DELP, PRECIP_RATE

logger = logging.getLogger(__name__)

cp = 1004
gravity = 9.81


def precipitation_sum(
    physics_precip: xr.DataArray, column_dq2: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return sum of physics precipitation and ML-induced precipitation. Output is
    thresholded to enforce positive precipitation.

    Args:
        physics_precip: precipitation from physics parameterizations [m]
        column_dq2: column-integrated moistening from ML [kg/m^2/s]
        dt: physics timestep [s]

    Returns:
        total precipitation [m]"""
    m_per_mm = 1 / 1000
    ml_precip = -column_dq2 * dt * m_per_mm  # type: ignore
    total_precip = physics_precip + ml_precip
    total_precip = total_precip.where(total_precip >= 0, 0)
    total_precip.attrs["units"] = "m"
    return total_precip


def precipitation_rate(
    precipitation_accumulation: xr.DataArray, dt: float
) -> xr.DataArray:
    """Return precipitation rate from a precipitation accumulation and timestep
    
    Args:
        precipitation_accumulation: precipitation accumulation [m]
        dt: timestep over which accumulation occurred [s]

    Returns:
        precipitation rate [kg/m^s/s]"""

    KG_PER_M2_PER_M = 1000.0
    precipitation_rate: xr.DataArray = (
        KG_PER_M2_PER_M * precipitation_accumulation / dt  # type: ignore
    )
    precipitation_rate.attrs["units"] = "kg/m^2/s"
    return precipitation_rate


def compute_ml_diagnostics(state: State, ml_tendency: State) -> Diagnostics:
    delp = state[DELP]
    dQ1 = ml_tendency.get("dQ1", xr.zeros_like(delp))
    dQ2 = ml_tendency.get("dQ2", xr.zeros_like(delp))
    net_moistening = (dQ2 * delp / gravity).sum("z")

    return dict(
        dQ1=dQ1.assign_attrs(units="K/s").assign_attrs(
            description="air temperature tendency due to ML"
        ),
        dQ2=dQ2.assign_attrs(units="kg/kg/s").assign_attrs(
            description="specific humidity tendency due to ML"
        ),
        air_temperature=state[TEMP],
        specific_humidity=state[SPHUM],
        pressure_thickness_of_atmospheric_layer=delp,
        net_moistening=(net_moistening)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated ML model moisture tendency"),
        net_heating=(dQ1 * delp / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated ML model heating"),
    )


def compute_ml_momentum_diagnostics(state: State, tendency: State) -> Diagnostics:
    delp = state[DELP]

    dQu = tendency.get("dQu", xr.zeros_like(delp))
    dQv = tendency.get("dQv", xr.zeros_like(delp))
    column_integrated_dQu = _mass_average(dQu, delp, "z")
    column_integrated_dQv = _mass_average(dQv, delp, "z")
    return dict(
        dQu=dQu.assign_attrs(units="m s^-2").assign_attrs(
            description="zonal wind tendency due to ML"
        ),
        dQv=dQv.assign_attrs(units="m s^-2").assign_attrs(
            description="meridional wind tendency due to ML"
        ),
        column_integrated_dQu=column_integrated_dQu.assign_attrs(
            units="m s^-2",
            description="column integrated zonal wind tendency due to ML",
        ),
        column_integrated_dQv=column_integrated_dQv.assign_attrs(
            units="m s^-2",
            description="column integrated meridional wind tendency due to ML",
        ),
    )


def rename_diagnostics(diags: Diagnostics):
    """Postfix ML output names with _diagnostic and create zero-valued outputs in
    their stead. Function operates in place."""
    ml_tendencies = {
        "net_moistening",
        "net_heating",
        "column_integrated_dQu",
        "column_integrated_dQv",
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
        "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
    }
    ml_tendencies_in_diags = ml_tendencies & set(diags)
    for variable in ml_tendencies_in_diags:
        attrs = diags[variable].attrs
        diags[f"{variable}_diagnostic"] = diags[variable].assign_attrs(
            description=attrs.get("description", "") + " (diagnostic only)"
        )
        diags[variable] = xr.zeros_like(diags[variable]).assign_attrs(attrs)


def compute_nudging_diagnostics(
    state: State, nudging_tendency: State, label: str = "_tendency_due_to_nudging"
) -> Diagnostics:
    """
    Compute diagnostic variables for nudging"""

    diags: Diagnostics = {}

    net_moistening = (
        (nudging_tendency[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated moistening due to nudging")
    )
    net_heating = (
        (nudging_tendency[TEMP] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated heating due to nudging")
    )

    diags.update(
        {
            "net_moistening_due_to_nudging": net_moistening,
            "net_heating_due_to_nudging": net_heating,
        }
    )

    if DELP in nudging_tendency.keys():
        net_mass_tendency = (
            (nudging_tendency[DELP] / gravity)
            .sum("z")
            .assign_attrs(
                units="kg/m^2/s",
                description="column-integrated mass tendency due to nudging",
            )
        )
        diags["net_mass_tendency_due_to_nudging"] = net_mass_tendency

    diags.update(_append_key_label(nudging_tendency, label))

    return diags


def _append_key_label(d: Diagnostics, suffix: str) -> Diagnostics:
    return_dict: Diagnostics = {}
    for key, value in d.items():
        return_dict[str(key) + suffix] = value
    return return_dict


def _mass_average(
    da: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    total_thickness = delp.sum(vertical_dim)
    mass_average = (da * delp).sum(vertical_dim) / total_thickness
    mass_average = mass_average.assign_attrs(**da.attrs)
    return mass_average


def compute_baseline_diagnostics(state: State) -> Diagnostics:

    return dict(
        water_vapor_path=(state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor"),
        physics_precip=(state[PRECIP_RATE])
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        ),
    )

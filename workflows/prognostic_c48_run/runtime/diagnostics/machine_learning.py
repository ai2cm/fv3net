from typing import MutableMapping, Hashable
import xarray as xr
import logging

logger = logging.getLogger(__name__)

TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
cp = 1004
gravity = 9.81

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


def compute_ml_diagnostics(state: State, ml_tendency: State) -> Diagnostics:

    physics_precip = state[PRECIP_RATE]
    delp = state[DELP]
    dQ1 = ml_tendency.get("dQ1", xr.zeros_like(delp))
    dQ2 = ml_tendency.get("dQ2", xr.zeros_like(delp))
    net_moistening = (dQ2 * delp / gravity).sum("z")

    return dict(
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
        water_vapor_path=(state[SPHUM] * delp / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor"),
        physics_precip=(physics_precip)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        ),
    )


def compute_ml_momentum_diagnostics(state: State, tendency: State) -> Diagnostics:
    delp = state[DELP]

    dQu = tendency.get("dQu", xr.zeros_like(delp))
    dQv = tendency.get("dQv", xr.zeros_like(delp))
    column_integrated_dQu = _mass_average(dQu, delp, "z")
    column_integrated_dQv = _mass_average(dQv, delp, "z")
    return dict(
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
    }
    ml_tendencies_in_diags = ml_tendencies & set(diags)
    for variable in ml_tendencies_in_diags:
        attrs = diags[variable].attrs
        diags[f"{variable}_diagnostic"] = diags[variable].assign_attrs(
            description=attrs["description"] + " (diagnostic only)"
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
    water_vapor_path = (
        (state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor")
    )
    physics_precip = (
        state[PRECIP_RATE]
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        )
    )

    diags.update(
        {
            "net_moistening_due_to_nudging": net_moistening,
            "net_heating_due_to_nudging": net_heating,
            "water_vapor_path": water_vapor_path,
            "physics_precip": physics_precip,
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

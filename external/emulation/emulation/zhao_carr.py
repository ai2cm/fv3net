"""Zhao Carr specific emulation options

the functions in this submodule know the variable names of the ZC microphysics

"""
import logging
import numpy as np
import numba
from fv3fit.emulation.transforms.zhao_carr import (
    CLASS_NAMES,
    ZERO_CLOUD,
    ZERO_TENDENCY,
    POSITIVE_TENDENCY,
    NEGATIVE_TENDENCY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "infer_gscond_cloud_from_conservation",
    "squash_gscond",
    "squash_precpd",
    "mask_where_fortran_cloud_identical",
    "mask_where_fortran_cloud_vanishes_gscond",
    "mask_zero_tend_classifier",
    "mask_zero_cloud_classifier",
    "mask_zero_cloud_classifier_precpd",
    "enforce_conservative_gscond",
    "enforce_conservative_phase_dependent",
]


# from physcons.f
GRAVITY = 9.80665  # m / s2
CP = 1.0046e3  # J / kg / K
LV = 2.5e6  # J / kg
RHO_WATER = 1000.0  # kg / m3


class Input:
    cloud_water = "cloud_water_mixing_ratio_input"
    humidity = "specific_humidity_input"
    temperature = "air_temperature_input"
    delp = "pressure_thickness_of_atmospheric_layer"


class GscondOutput:
    cloud_water = "cloud_water_mixing_ratio_after_gscond"
    humidity = "specific_humidity_after_gscond"
    temperature = "air_temperature_after_gscond"


class PrecpdOutput:
    cloud_water = "cloud_water_mixing_ratio_after_precpd"
    humidity = "specific_humidity_after_precpd"
    temperature = "air_temperature_after_precpd"
    precip = "total_precipitation"


def squash_water_water_conserving(cloud, humidity, bound: float):
    cloud_out = np.where(cloud < bound, 0, cloud)
    qv_out = humidity + (cloud - cloud_out)
    return cloud_out, qv_out


def _apply_squash(struct, output_state, cloud_squash: float):
    out = {**output_state}
    if struct.cloud_water in output_state:
        cloud, humidity = squash_water_water_conserving(
            output_state[struct.cloud_water],
            output_state[struct.humidity],
            cloud_squash,
        )
        out[struct.cloud_water] = cloud
        out[struct.humidity] = humidity
    return out


def infer_gscond_cloud_from_conservation(state, emulator):
    out = {**emulator}
    humidity_change = emulator[GscondOutput.humidity] - state[Input.humidity]
    out[GscondOutput.cloud_water] = state[Input.cloud_water] - humidity_change
    return out


def squash_gscond(state, emulator, cloud_squash):
    return _apply_squash(GscondOutput, emulator, cloud_squash)


def squash_precpd(state, emulator, cloud_squash):
    return _apply_squash(PrecpdOutput, emulator, cloud_squash)


def _limit_net_condensation_conserving(state, net_condensation):
    available_vapor = state[Input.humidity]
    available_liquid = state[Input.cloud_water]
    condensation = np.where(net_condensation > 0, net_condensation, 0.0)
    evaporation = np.where(net_condensation < 0, net_condensation, 0.0)

    limited_evaporation = np.maximum(evaporation, -available_liquid)
    limited_condensation = np.minimum(condensation, available_vapor)
    logger.info({"evap_mass_cons_adj": np.sum(limited_evaporation - evaporation)})
    logger.info({"cond_mass_cons_adj": np.sum(limited_condensation - condensation)})
    net_condensation = limited_evaporation + limited_condensation
    return net_condensation


def apply_condensation_liquid_phase(state, net_condensation):
    # from physcons.f
    lv = 2.5e6
    return apply_condensation(state, net_condensation, lv=lv)


@numba.njit
def ice_water_flag(temperature_celsius, cloud):
    """Implement ice water id number from gscond.f

    If this is 1 then the cloud is all ice. If it is 0, then it is all liquid.

    Note a small difference in < -15 case to remove the RH dependent threshold
    """

    n, z = temperature_celsius.shape
    iw = np.zeros_like(temperature_celsius)
    climit = 1e-20
    # loop over slow coordinate
    for i in range(n):
        for k in range(z - 1, -1, -1):
            t_celsius = temperature_celsius[i, k]
            if t_celsius < -15:
                # no RH dependent threshold here.
                iw[i, k] = 1.0
            elif t_celsius > 0.0:
                iw[i, k] = 0.0
            else:
                if k < z - 1 and iw[i, k + 1] == 1 and cloud[i, k] > climit:
                    iw[i, k] = 1.0
    return iw


def latent_heat_phase_dependent(iw):
    hvap = 2.5e6
    hfus = 3.3358e5
    return hvap + iw * hfus


def apply_condensation_phase_dependent(state, net_condensation):
    temperature_celsius = state[Input.temperature] - 273.16
    iw = ice_water_flag(temperature_celsius, cloud=state[Input.cloud_water])
    lv = latent_heat_phase_dependent(iw)
    return apply_condensation(state, net_condensation=net_condensation, lv=lv)


def apply_condensation(state, net_condensation, lv):
    # from physcons.f
    cp = 1.0046e3
    cloud_out = state[Input.cloud_water] + net_condensation
    qv_out = state[Input.humidity] - net_condensation
    latent_heating = lv * net_condensation / cp
    temperature_out = state[Input.temperature] + latent_heating
    return {
        GscondOutput.cloud_water: cloud_out,
        GscondOutput.humidity: qv_out,
        GscondOutput.temperature: temperature_out,
    }


def _update_with_net_condensation(cloud_out, state, emulator):
    net_condensation = cloud_out - state[Input.cloud_water]
    net_condensation = _limit_net_condensation_conserving(state, net_condensation)
    return {**emulator, **apply_condensation_liquid_phase(state, net_condensation)}


def mask_where_fortran_cloud_vanishes_gscond(state, emulator):
    threshold = 1e-15
    cloud_out = np.where(
        state[GscondOutput.cloud_water] < threshold,
        0,
        emulator[GscondOutput.cloud_water],
    )
    return _update_with_net_condensation(cloud_out, state, emulator)


def mask_where_fortran_cloud_identical(state, emulator):
    cloud_out = np.where(
        state[GscondOutput.cloud_water] == state[Input.cloud_water],
        state[Input.cloud_water],
        emulator[GscondOutput.cloud_water],
    )
    return _update_with_net_condensation(cloud_out, state, emulator)


def _get_classify_output(logit_classes, one_hot_axis=0):
    names = sorted(CLASS_NAMES)
    one_hot = logit_classes == np.max(logit_classes, axis=one_hot_axis, keepdims=True)
    d = {name: np.take(one_hot, i, one_hot_axis) for i, name in enumerate(names)}
    d["nontrivial_tendency"] = d[POSITIVE_TENDENCY] | d[NEGATIVE_TENDENCY]
    return d


def mask_zero_cloud_classifier(state, emulator):
    cloud_out = np.where(
        _get_classify_output(emulator["gscond_classes"])[ZERO_CLOUD],
        0,
        emulator[GscondOutput.cloud_water],
    )
    return _update_with_net_condensation(cloud_out, state, emulator)


def mask_zero_tend_classifier(state, emulator):
    cloud_out = np.where(
        _get_classify_output(emulator["gscond_classes"])[ZERO_TENDENCY],
        state[Input.cloud_water],
        emulator[GscondOutput.cloud_water],
    )
    return _update_with_net_condensation(cloud_out, state, emulator)


def mask_zero_cloud_classifier_precpd(state, emulator):
    cloud_out = np.where(
        _get_classify_output(emulator["precpd_classes"])[ZERO_CLOUD],
        0,
        emulator[PrecpdOutput.cloud_water],
    )
    out = {**emulator, PrecpdOutput.cloud_water: cloud_out}
    return out


def enforce_conservative_gscond(state, emulator):
    cloud_out = emulator[GscondOutput.cloud_water]
    return _update_with_net_condensation(cloud_out, state, emulator)


def enforce_conservative_phase_dependent(state, emulator):
    cloud_out = emulator[GscondOutput.cloud_water]
    net_condensation = cloud_out - state[Input.cloud_water]
    net_condensation = _limit_net_condensation_conserving(state, net_condensation)
    return {**emulator, **apply_condensation_phase_dependent(state, net_condensation)}


def mixing_ratio_to_mass(x: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """convert data proportional to kg/kg -> kg/m2"""
    return x * delp / GRAVITY


def mass_to_mixing_ratio(x: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """convert data proportional to kg/m2 -> kg/kg"""
    return x / delp * GRAVITY


def liquid_water_equivalent(x: np.ndarray) -> np.ndarray:
    """convert data proportional to kg/m2 -> m"""
    return x / RHO_WATER


def _strict_conservative_precip_from_TOA_to_surface(
    condensate_to_precip: np.ndarray, precip_to_vapor: np.ndarray
):

    """
    Iterates backwards through precip source and evaporation terms to
    determine surface precip.  Function limits evaporation to available
    precipitation at each level, and limits source/sink terms to be positive.

    Expects data in [feature x sample] dimensions (default for fields from Fortran)
    """

    limited_c_to_p = np.maximum(condensate_to_precip, 0)  # no condensate from precip
    limited_p_to_v = np.maximum(precip_to_vapor, 0)  # no precip from vapor

    if not len(precip_to_vapor.shape) == 2:
        raise ValueError(
            "Expected 2D inputs to the strict conservative precip function"
        )

    num_features, num_samples = precip_to_vapor.shape

    total_precip = np.zeros(num_samples)

    # calculate precip and evaporation starting from TOA
    for k in range(num_features - 1, -1, -1):
        precip = limited_c_to_p[k]
        total_precip += precip
        evaporation = limited_p_to_v[k]
        limited_evap = np.minimum(total_precip, evaporation)
        total_precip -= limited_evap
        limited_p_to_v[k, :] = limited_evap

    return limited_c_to_p, limited_p_to_v, total_precip


def enforce_conservative_precpd(state, emulator):
    cloud_change = emulator[PrecpdOutput.cloud_water] - state[GscondOutput.cloud_water]
    humidity_change = emulator[PrecpdOutput.humidity] - state[GscondOutput.humidity]

    # switch to kg / m2
    delp = state[Input.delp]
    precip_source = mixing_ratio_to_mass(-1 * cloud_change, delp)
    precip_sink = mixing_ratio_to_mass(humidity_change, delp)

    [
        precip_src_limited,
        precip_sink_limited,
        total_precip,
    ] = _strict_conservative_precip_from_TOA_to_surface(precip_source, precip_sink)

    surface_precip_m = liquid_water_equivalent(total_precip)
    limited_evaporation = mass_to_mixing_ratio(precip_sink_limited, delp)

    # temperature adjust
    evaporative_cooling = LV / CP * -1 * limited_evaporation

    cloud_out = state[GscondOutput.cloud_water] + mass_to_mixing_ratio(
        -1 * precip_src_limited, delp
    )
    humidity_out = state[GscondOutput.humidity] + limited_evaporation
    temperature_out = state[GscondOutput.temperature] + evaporative_cooling

    return {
        **emulator,
        PrecpdOutput.cloud_water: cloud_out,
        PrecpdOutput.humidity: humidity_out,
        PrecpdOutput.temperature: temperature_out,
        PrecpdOutput.precip: surface_precip_m,
    }


def conservative_precip_simple(state, emulator, sum_axis=0):
    qv_before = state[GscondOutput.humidity]
    qv_after = emulator[PrecpdOutput.humidity]
    qc_before = state[GscondOutput.cloud_water]
    qc_after = emulator[PrecpdOutput.cloud_water]
    delp = state[Input.delp]

    water_before = qv_before + qc_before
    water_after = qv_after + qc_after
    column_water_before = np.sum(
        mixing_ratio_to_mass(water_before, delp), axis=sum_axis
    )
    column_water_after = np.sum(mixing_ratio_to_mass(water_after, delp), axis=sum_axis)
    surface_precipitation = liquid_water_equivalent(
        column_water_before - column_water_after
    )
    return {**emulator, PrecpdOutput.precip: surface_precipitation}


def preserve_state_inputs(state, emulator):
    input_fields = {k: v for k, v in state.items() if "_input" in k}
    logger.info(f"Preserving input state fields: {list(input_fields.keys())}")
    return {**emulator, **input_fields}

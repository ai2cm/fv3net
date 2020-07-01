import logging
import numpy as np

import vcm

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400


def calc_diagnostics(prognostic, verification, grid):
    """
    Diagnostic function for calculating all diurnal cycle
    information from physics output.
    """

    logger.info("Preparing diurnal cycle diagnostics")

    diurnal_cycle_vars = [
        "column_integrated_dQ1",
        "column_integrated_pQ1",
        "column_integrated_Q1",
        "column_integrated_dQ2",
        "column_integrated_pQ2",
        "column_integrated_Q2",
        "PRATEsfc",
        "LHTFLsfc",
        "SLMSKsfc",
    ]

    prognostic = prognostic[[var for var in diurnal_cycle_vars if var in prognostic]]

    prognostic["lon"] = grid["lon"]
    prog_diurnal_ds = _calc_ds_diurnal_cycle(prognostic)

    verif = verification[[var for var in diurnal_cycle_vars if var in verification]]
    verif["lon"] = grid["lon"]
    verif_diurnal_ds = _calc_ds_diurnal_cycle(verif)

    prog_diurnal_ds = _add_diurnal_moisture_components(prog_diurnal_ds)
    verif_diurnal_ds = _add_diurnal_moisture_components(verif_diurnal_ds)

    prog_diurnal_ds = _add_diurn_comparison(prog_diurnal_ds, verif_diurnal_ds)

    return prog_diurnal_ds


def _calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle for all variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time = vcm.local_time(ds, time="time", lon_var="lon")
    local_time.attrs = {"long_name": "local time", "units": "hour"}

    local_time = np.floor(local_time)  # equivalent to hourly binning
    ds["local_time"] = local_time
    diurnal_ds = ds.groupby("local_time").mean()

    return diurnal_ds


def _add_diurnal_moisture_components(diurnal_ds):

    # pq2 is E - P in mm/day
    precip = diurnal_ds["physics_precip"] * SECONDS_PER_DAY
    diurnal_ds["diurn_comp_P"] = precip
    evap = vcm.latent_heat_flux_to_evaporation(diurnal_ds["LHTFLsfc"]) * SECONDS_PER_DAY
    diurnal_ds["diurn_comp_E"] = evap

    if "column_integrated_dQ2" in diurnal_ds:
        dQ2 = diurnal_ds["column_integrated_dQ2"]
        diurnal_ds["diurn_comp_-dQ2"] = -dQ2
        precip_phys_ml = precip - dQ2
        precip_phys_ml.attrs = {
            "long_name": "Total precipitation (P - dQ2)",
            "units": "mm/day",
        }
        diurnal_ds["diurn_comp_P-dQ2"] = precip_phys_ml

    return diurnal_ds


def _add_diurn_comparison(prognostic_diurnal, verif_diurnal):

    evap_compare = (
        prognostic_diurnal["diurn_comp_E"] - verif_diurnal["diurn_comp_E"]
    )
    evap_compare.attrs = {
        "long_name": "Evaporation diurnal cycle difference [coarse - hires]",
        "units": "mm/day",
    }
    prognostic_diurnal["evap_against_verif"] = evap_compare

    if "diurn_comp_P-dQ2" in prognostic_diurnal:
        prognostic_precip = prognostic_diurnal["diurn_comp_P-dQ2"]
    else:
        prognostic_precip = prognostic_diurnal["diurn_comp_P"]

    precip_compare = prognostic_precip - verif_diurnal["diurn_comp_P"]
    precip_compare.attrs = {
        "long_name": (
            "Precipitation diurnal cycle difference (includes ML) [coarse - hires]"
        ),
        "units": "mm/day",
    }
    prognostic_diurnal["precip_against_verif"] = precip_compare

    net_precip_compare = precip_compare - evap_compare
    net_precip_compare.attrs = {
        "long_name": (
            "Net precip diurnal cycle difference (includes ML) [coarse - hires]"
        ),
        "units": "mm/day",
    }
    prognostic_diurnal["net_precip_against_verif"] = net_precip_compare

    return prognostic_diurnal

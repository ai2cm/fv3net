import logging
import numpy as np
import xarray as xr

import vcm

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400


def calc_diagnostics(prognostic, verification, grid):
    """
    Diagnostic function for calculating all diurnal cycle
    information from physics output.
    """

    logger.info("Preparing diurnal cycle diagnostics")

    prog_diurnal_ds = _calc_ds_diurnal_cycle(prognostic.assign(lon=grid["lon"]))
    verif_diurnal_ds = _calc_ds_diurnal_cycle(verification.assign(lon=grid["lon"]))

    prog_diurnal_ds = _add_diurnal_moisture_components(prog_diurnal_ds)
    verif_diurnal_ds = _add_diurnal_moisture_components(verif_diurnal_ds)

    prog_diurnal_ds = _add_diurn_bias(prog_diurnal_ds, verif_diurnal_ds)

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
    diurnal_cycles = ds.groupby("local_time").mean()

    return diurnal_cycles


def _add_diurnal_moisture_components(diurnal_cycles: xr.Dataset):
    """
    Add individual moisture components for diurnal cycle plots with a long-name
    and attributes.  The naming is used by report generation to determine the
    component shorthand.  E.g., diurn_comp_<component_name>
    """

    evap = vcm.latent_heat_flux_to_evaporation(diurnal_cycles["LHTFLsfc"])
    evap *= SECONDS_PER_DAY
    evap.attrs = {"long_name": "Evaporation", "units": "mm/day"}
    diurnal_cycles["diurn_comp_E"] = evap

    precip = diurnal_cycles["PRATEsfc"] * SECONDS_PER_DAY
    precip.attrs = {"long_name": "Physics precipitation", "units": "mm/day"}
    diurnal_cycles["diurn_comp_P"] = precip

    # no dQ2 in verification data
    if "column_integrated_dQ2" in diurnal_cycles:
        dQ2 = diurnal_cycles["column_integrated_dQ2"]
        diurnal_cycles["diurn_comp_-dQ2"] = -dQ2
        diurnal_cycles["diurn_comp_-dQ2"].attrs = {
            "long_name": "<-dQ2> column integrated drying from ML",
            "units": "mm/day",
        }

        precip_phys_ml = precip - dQ2
        precip_phys_ml.attrs = {
            "long_name": "Total precipitation (P - dQ2)",
            "units": "mm/day",
        }
        diurnal_cycles["diurn_comp_P-dQ2"] = precip_phys_ml

    return diurnal_cycles


def _add_diurn_bias(prognostic_diurnal, verif_diurnal):
    """
    Add comparisons of diurnal cycle against verification data for plotting
    """

    evap_compare = prognostic_diurnal["diurn_comp_E"] - verif_diurnal["diurn_comp_E"]
    evap_compare.attrs = {
        "long_name": "Evaporation diurnal cycle bias [coarse - hires]",
        "units": "mm/day",
    }
    prognostic_diurnal["evap_against_verif"] = evap_compare

    prognostic_precip = prognostic_diurnal["diurn_comp_P-dQ2"]
    precip_compare = prognostic_precip - verif_diurnal["diurn_comp_P"]
    precip_compare.attrs = {
        "long_name": ("Precipitation (P-<dQ2>) diurnal cycle bias [coarse - hires]"),
        "units": "mm/day",
    }
    prognostic_diurnal["precip_against_verif"] = precip_compare

    net_precip_compare = precip_compare - evap_compare
    net_precip_compare.attrs = {
        "long_name": ("Net precip (-<Q2>) diurnal cycle bias [coarse - hires]"),
        "units": "mm/day",
    }
    prognostic_diurnal["net_precip_against_verif"] = net_precip_compare

    return prognostic_diurnal

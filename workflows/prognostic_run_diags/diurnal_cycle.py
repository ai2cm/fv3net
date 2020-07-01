import logging
import numpy as np

import vcm

logger = logging.getLogger(__name__)


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
    diurnal_ds = _calc_ds_diurnal_cycle(prognostic)

    return diurnal_ds


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

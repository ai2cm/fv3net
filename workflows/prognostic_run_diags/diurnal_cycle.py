import logging
import numpy as np

import vcm

logger = logging.getLogger(__name__)


def diurnal_cycles(prognostic, verification, grid):

    logger.info("Preparing diurnal cycle diagnostics")

    diurnal_cycle_vars = [
        f"column_integrated_{a}Q{b}" for a in ["d", "p", ""] for b in ["1", "2"]
    ] + ["PRATEsfc", "LHTFLsfc", "SLMSKsfc"]
    prognostic = prognostic[[var for var in diurnal_cycle_vars if var in prognostic]]

    prognostic["lon"] = grid["lon"]
    diurnal_ds = _calc_ds_diurnal_cycle(prognostic)
    
    return diurnal_ds


def _calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle on moisture variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time = vcm.local_time(ds, time="time", lon_var="lon")
    local_time.attrs = {"long_name": "local time", "units": "hour"}

    local_time = np.floor(local_time)  # equivalent to hourly binning
    ds["local_time"] = local_time
    diurnal_ds = ds.groupby("local_time").mean()

    return diurnal_ds

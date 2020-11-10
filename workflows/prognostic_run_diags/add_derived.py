import logging
import xarray as xr

import vcm

SECONDS_PER_DAY = 86400

logger = logging.getLogger(__name__)


def physics_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute selected derived variables from a physics dataset
    and merge them back in.
    
    Args:
        ds: Dataset to calculated derived values from and merge to
    """
    arrays = []
    for func in [
        _column_pq1,
        _column_pq2,
        _column_dq1,
        _column_dq2,
        _column_q1,
        _column_q2,
        _total_precip,
    ]:
        try:
            arrays.append(func(ds))
        except (KeyError, AttributeError):  # account for ds[var] and ds.var notations
            logger.warning(f"Missing variable for calculation in {func.__name__}")
    return ds.merge(xr.merge(arrays))


def _column_pq1(ds: xr.Dataset) -> xr.DataArray:
    net_heating_arg_labels = [
        "DLWRFsfc",
        "DSWRFsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "USWRFsfc",
        "USWRFtoa",
        "DSWRFtoa",
        "SHTFLsfc",
        "PRATEsfc",
    ]
    net_heating_args = [ds[var] for var in net_heating_arg_labels]
    column_pq1 = vcm.net_heating(*net_heating_args)
    column_pq1.attrs = {
        "long_name": "<pQ1> column integrated heating from physics",
        "units": "W/m^2",
    }
    return column_pq1.rename("column_integrated_pQ1")


def _column_pq2(ds: xr.Dataset) -> xr.DataArray:
    evap = vcm.latent_heat_flux_to_evaporation(ds.LHTFLsfc)
    column_pq2 = SECONDS_PER_DAY * (evap - ds.PRATEsfc)
    column_pq2.attrs = {
        "long_name": "<pQ2> column integrated moistening from physics",
        "units": "mm/day",
    }
    return column_pq2.rename("column_integrated_pQ2")


def _column_dq1(ds: xr.Dataset) -> xr.DataArray:
    if "net_heating" in ds:
        column_dq1 = ds.net_heating
    else:
        # assume given dataset is for a baseline or verification run
        column_dq1 = xr.zeros_like(ds.PRATEsfc)
    column_dq1.attrs = {
        "long_name": "<dQ1> column integrated heating from ML",
        "units": "W/m^2",
    }
    return column_dq1.rename("column_integrated_dQ1")


def _column_dq2(ds: xr.Dataset) -> xr.DataArray:
    if "net_moistening" in ds:
        column_dq2 = SECONDS_PER_DAY * ds.net_moistening
    elif "column_moistening_nudge" in ds:
        # name for column integrated humidity nudging in nudge-to-obs runs
        column_dq2 = SECONDS_PER_DAY * ds.column_moistening_nudge
    else:
        # assume given dataset is for a baseline run or a verification run
        # without humidity nudging
        column_dq2 = xr.zeros_like(ds.PRATEsfc)
    column_dq2.attrs = {
        "long_name": "<dQ2> column integrated moistening from ML",
        "units": "mm/day",
    }
    return column_dq2.rename("column_integrated_dQ2")


def _column_q1(ds: xr.Dataset) -> xr.DataArray:
    column_q1 = _column_pq1(ds) + _column_dq1(ds)
    column_q1.attrs = {
        "long_name": "<Q1> column integrated heating from physics+ML",
        "units": "W/m^2",
    }
    return column_q1.rename("column_integrated_Q1")


def _column_q2(ds: xr.Dataset) -> xr.DataArray:
    column_q2 = _column_pq2(ds) + _column_dq2(ds)
    column_q2.attrs = {
        "long_name": "<Q2> column integrated moistening from physics+ML",
        "units": "mm/day",
    }
    return column_q2.rename("column_integrated_Q2")


def _total_precip(ds: xr.Dataset) -> xr.DataArray:
    total_precip = ds.PRATEsfc * SECONDS_PER_DAY - _column_dq2(ds)
    total_precip.attrs = {
        "long_name": "P - <dQ2> total precipitation",
        "units": "mm/day",
    }
    return total_precip.rename("total_precip")

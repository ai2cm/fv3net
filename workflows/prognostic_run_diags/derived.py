import logging
import xarray as xr
from collections import defaultdict
from toolz import curry
from typing import Callable, Dict

import vcm
import save_prognostic_run_diags

DiagArg = save_prognostic_run_diags.DiagArg

_DERIVED_VAR_FNS = defaultdict(Callable)
SECONDS_PER_DAY = 86400

logger = logging.getLogger(__name__)


@curry
def add_to_derived_vars(diags_key: str, func: Callable[[xr.Dataset], xr.Dataset]):
    """Add a function that computes additional derived variables from input datasets.
    These derived variables should be simple combinations of input variables, not
    reductions such as global means which are diagnostics to be computed later.

    Args:
        diags_key: a key for a group of inputs/diagnostics
        func: a function which adds new variables to a given dataset
    """
    _DERIVED_VAR_FNS[diags_key] = func


def compute_all_derived_vars(input_datasets: Dict[str, DiagArg]) -> Dict[str, DiagArg]:
    """Compute derived variables for all input data sources.

    Args:
        input_datasets: Input datasets with keys corresponding to the appropriate group
        of inputs/diagnostics.

    Returns:
        input datasets with derived variables added to prognostic and verification data
    """
    for key, func in _DERIVED_VAR_FNS.items():
        prognostic, verification, grid = input_datasets[key]
        logger.info(f"Preparing all derived variables for {key} prognostic data")
        prognostic = prognostic.merge(func(prognostic))
        logger.info(f"Preparing all derived variables for {key} verification data")
        verification = verification.merge(func(verification))
        input_datasets[key] = prognostic, verification, grid
    return input_datasets


@add_to_derived_vars("physics")
def derived_physics_variables(ds: xr.Dataset) -> xr.Dataset:
    """Compute derived variables for physics datasets"""
    arrays = []
    for func in [
        _column_pq1,
        _column_pq2,
        _column_dq1,
        _column_dq2,
        _column_q1,
        _column_q2,
    ]:
        try:
            arrays.append(func(ds))
        except (KeyError, AttributeError):  # account for ds[var] and ds.var notations
            logger.warning(f"Missing variable for calculation in {func.__name__}")
    return xr.merge(arrays)


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


def _column_pq2(ds: xr.Dataset) -> xr.Dataset:
    evap = vcm.latent_heat_flux_to_evaporation(ds.LHTFLsfc)
    column_pq2 = SECONDS_PER_DAY * (evap - ds.PRATEsfc)
    column_pq2.attrs = {
        "long_name": "<pQ2> column integrated moistening from physics",
        "units": "mm/day",
    }
    return column_pq2.rename("column_integrated_pQ2")


def _column_dq1(ds: xr.Dataset) -> xr.Dataset:
    column_dq1 = ds.net_heating
    column_dq1.attrs = {
        "long_name": "<dQ1> column integrated heating from ML",
        "units": "W/m^2",
    }
    return column_dq1.rename("column_integrated_dQ1")


def _column_dq2(ds: xr.Dataset) -> xr.Dataset:
    column_dq2 = SECONDS_PER_DAY * ds.net_moistening
    column_dq2.attrs = {
        "long_name": "<dQ2> column integrated moistening from ML",
        "units": "mm/day",
    }
    return column_dq2.rename("column_integrated_dQ2")


def _column_q1(ds: xr.Dataset) -> xr.Dataset:
    column_q1 = _column_pq1(ds) + _column_dq1(ds)
    column_q1.attrs = {
        "long_name": "<Q1> column integrated heating from physics+ML",
        "units": "W/m^2",
    }
    return column_q1.rename("column_integrated_Q1")


def _column_q2(ds: xr.Dataset) -> xr.Dataset:
    column_q2 = _column_pq2(ds) + _column_dq2(ds)
    column_q2.attrs = {
        "long_name": "<Q2> column integrated moistening from physics+ML",
        "units": "mm/day",
    }
    return column_q2.rename("column_integrated_Q2")

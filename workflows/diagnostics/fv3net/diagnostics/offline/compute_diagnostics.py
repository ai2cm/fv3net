from fv3net.diagnostics._shared.registry import Registry, prepare_diag_dict
import fv3net.diagnostics._shared.transform as transform
from fv3net.diagnostics._shared.constants import DiagArg, SURFACE_TYPE
import logging
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict, Union, Mapping, Optional
import xarray as xr

from vcm import thermo, safe, weighted_average, local_time


logger = logging.getLogger(__name__)

DOMAINS = (
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
)
SURFACE_TYPE_ENUMERATION = {0.0: "sea", 1.0: "land", 2.0: "sea"}
DERIVATION_DIM = "derivation"


def merge_diagnostics(metrics: Sequence[Tuple[str, xr.Dataset]]):
    out: Dict[str, xr.DataArray] = {}
    for (name, ds) in metrics:
        out.update(prepare_diag_dict(name, ds))
    # ignoring type error that complains if Dataset created from dict
    return xr.Dataset(out)  # type: ignore


diagnostics_registry = Registry(merge_diagnostics)


def compute_diagnostics(
    prediction: xr.Dataset,
    target: xr.Dataset,
    grid: xr.Dataset,
    delp: xr.DataArray,
    n_jobs: int = -1,
):
    diag_arg = DiagArg(prediction, target, grid, delp=delp)
    return diagnostics_registry.compute(diag_arg, n_jobs=n_jobs)


def conditional_average_over_domain(
    ds: xr.Dataset,
    grid: xr.Dataset,
    primary_vars: Sequence[str],
    domain: str,
    net_precipitation: Optional[xr.DataArray] = None,
    uninformative_coords: Sequence[str] = ["tile", "z", "y", "x"],
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds: xarray datasets with relevant variables batched in time
        grid: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
        domains: sequence of area domains over which to produce conditional
            averages; optional, defaults to global, land, sea, and positive and
            negative net_precipitation domains
        primary_vars: sequence of variables for which to compute column integrals
            and composite means; optional, defaults to dQs, pQs and Qs
        net_precipitation: xr.DataArray of net_precipitation values for computing
            composites, typically supplied by SHiELD net_precipitation; optional
        uninformative_coords: sequence of names of uninformative (i.e.,
            range(len(dim))), coordinates to be dropped
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = ds.drop_vars(names=uninformative_coords, errors="ignore")

    grid = grid.drop_vars(names=uninformative_coords, errors="ignore")
    surface_type_array = _snap_mask_to_type(grid[SURFACE_TYPE])
    if "net_precipitation" in domain:
        net_precipitation_type_array = _snap_net_precipitation_to_type(
            net_precipitation
        )
        cell_type = net_precipitation_type_array.drop_vars(
            names=uninformative_coords, errors="ignore"
        )
    else:
        cell_type = surface_type_array
    domain_average = _conditional_average(
        safe.get_variables(ds, primary_vars), cell_type, domain, grid["area"],
    )
    return domain_average.mean("time")


def _conditional_average(
    ds: Union[xr.Dataset, xr.DataArray],
    cell_type_array: xr.DataArray,
    category: str,
    area: xr.DataArray,
    dims: Sequence[str] = ["tile", "y", "x"],
) -> xr.Dataset:
    """Average over a conditional type
    
    Args:
        ds: xr dataarray or dataset of variables to averaged conditionally
        cell_type_array: xr datarray of cell category strings
        category: str of category over which to conditionally average
        area: xr datarray of grid cell areas for weighted averaging
        dims: dimensions to average over
            
    Returns:
        xr dataarray or dataset of conditionally averaged variables
    """

    if category == "global":
        area_masked = area
    elif category in DOMAINS:
        area_masked = area.where(cell_type_array == category)
    else:
        raise ValueError(
            f"surface type {category} not in provided surface type array "
            f"with types {DOMAINS}."
        )

    return weighted_average(ds, area_masked, dims)


def _snap_mask_to_type(
    float_mask: xr.DataArray,
    enumeration: Mapping[float, str] = SURFACE_TYPE_ENUMERATION,
    atol: float = 1e-7,
) -> xr.DataArray:
    """Convert float surface type array to categorical surface type array
    
    Args:
        float_mask: xr dataarray of float cell types
        enumeration: mapping of surface type str names to float values
        atol: absolute tolerance of float value matching
            
    Returns:
        types: xr dataarray of str categorical cell types
    
    """

    types = np.full(float_mask.shape, np.nan)
    for type_number, type_name in enumeration.items():
        types = np.where(
            np.isclose(float_mask.values, type_number, atol), type_name, types
        )

    types = xr.DataArray(types, float_mask.coords, float_mask.dims)

    return types


def _snap_net_precipitation_to_type(
    net_precipitation: xr.DataArray, type_names: Mapping[str, str] = None
) -> xr.DataArray:
    """Convert net_precipitation array to positive and negative categorical types
    
    Args:
        net_precipitation: xr.DataArray of numerical values
        type_names: Mapping relating the "positive" and "negative" cases to their
            categorical type names
            
    Returns:
        types: xr dataarray of categorical str type
    
    """

    type_names = type_names or {
        "negative": "negative_net_precipitation",
        "positive": "positive_net_precipitation",
    }

    if any(key not in type_names for key in ["negative", "positive"]):
        raise ValueError("'type_names' must include 'positive' and negative' as keys")

    types = np.where(
        net_precipitation.values < 0, type_names["negative"], type_names["positive"]
    )

    return xr.DataArray(types, net_precipitation.coords, net_precipitation.dims)


def _calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle for all variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time_ = local_time(ds, time="time", lon_var="lon")
    local_time_.attrs = {"long_name": "local time", "units": "hour"}
    ds["local_time"] = np.floor(local_time_)  # equivalent to hourly binning
    with xr.set_options(keep_attrs=True):
        diurnal_cycles = ds.drop("lon").groupby("local_time").mean()
    return diurnal_cycles


def _predicts_sphum_tendency(ds):
    for var in ds.data_vars:
        if "q2" in var.lower():
            return True
    return False


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"{mask_type}_diurnal_cycle")
    @transform.apply(transform.select_2d_variables)
    def diurnal_cycle(diag_arg, mask_type=mask_type):
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        ds = xr.concat(
            [predicted, target],
            dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
        )
        ds["lon"] = grid["lon"]
        return _calc_ds_diurnal_cycle(ds)


for domain in ["positive_net_precipitation", "negative_net_precipitation"]:

    @diagnostics_registry.register(f"{domain}_time_mean")
    @transform.apply(transform.subset_variables, ["dQ1", "dQ2", "Q2"])
    def precip_domain_time_mean(diag_arg, domain=domain):
        logger.info(f"Preparing conditional averages over domain {domain}")
        predicted, target, grid, delp = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
            diag_arg.delp,
        )
        if len(predicted) > 0 and _predicts_sphum_tendency(predicted):
            net_precip = thermo.minus_column_integrated_moistening(target["Q2"], delp)
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            domain_avg = conditional_average_over_domain(
                ds,
                grid,
                predicted.data_vars,
                domain=domain,
                net_precipitation=net_precip,
            )
            return domain_avg
        else:
            return xr.Dataset()


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"{mask_type}_time_mean")
    def sfc_type_domain_mean(diag_arg, domain=mask_type):
        logger.info(f"Preparing conditional averages over domain {domain}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            domain_avg = conditional_average_over_domain(
                ds, grid, predicted.data_vars, domain=domain
            )
            return domain_avg


@diagnostics_registry.register("time_mean")
def time_mean(diag_arg):
    logger.info(f"Preparing time means")
    predicted, target = diag_arg.prediction, diag_arg.verification
    ds = xr.concat(
        [predicted, target], dim=pd.Index(["predict", "target"], name=DERIVATION_DIM)
    )
    return ds.mean("time")

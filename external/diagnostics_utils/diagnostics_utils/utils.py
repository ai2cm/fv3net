from .config import (
    VARNAMES,
    SURFACE_TYPE_ENUMERATION,
    DOMAINS,
    PRIMARY_VARS,
)
from vcm import thermo, safe, mass_integrate, weighted_average
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping, Union, Optional

logger = logging.getLogger(__name__)

UNITS = {
    "column_integrated_dq1": "[W/m2]",
    "column_integrated_dq2": "[mm/day]",
    "column_integrated_q1": "[W/m2]",
    "column_integrated_q2": "[mm/day]",
    "column_integrated_dqu": "[Pa]",
    "column_integrated_dqv": "[Pa]",
    "dq1": "[K/s]",
    "pq1": "[K/s]",
    "q1": "[K/s]",
    "dq2": "[kg/kg/s]",
    "pq2": "[kg/kg/s]",
    "q2": "[kg/kg/s]",
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface": "[W/m2]",
    "net_shortwave_sfc_flux_derived": "[W/m2]",
}
UNITS.update({f"error_in_{var}": UNITS[var] for var in UNITS})
UNITS.update({f"{var}_snapshot": UNITS[var] for var in UNITS})


def reduce_to_diagnostic(
    ds: xr.Dataset,
    grid: xr.Dataset,
    domains: Sequence[str] = DOMAINS,
    primary_vars: Sequence[str] = PRIMARY_VARS,
    net_precipitation: Optional[xr.DataArray] = None,
    time_dim: str = "time",
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
        time_dim: name of the dataset time dimension to average over; optional,
            defaults to 'time'
        uninformative_coords: sequence of names of uninformative (i.e.,
            range(len(dim))), coordinates to be dropped
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = ds.drop_vars(names=uninformative_coords, errors="ignore")
    ds = _rechunk_time_z(ds)

    grid = grid.drop_vars(names=uninformative_coords, errors="ignore")
    surface_type_array = snap_mask_to_type(grid[VARNAMES["surface_type"]])
    if net_precipitation is None:
        domains = [domain for domain in domains if "net_precipitation" not in domain]
    if any(["net_precipitation" in category for category in domains]):
        net_precipitation_type_array = snap_net_precipitation_to_type(net_precipitation)
        net_precipitation_type_array = net_precipitation_type_array.drop_vars(
            names=uninformative_coords, errors="ignore"
        )

    domain_datasets = {}
    for category in domains:
        varname = f"{category}_average"
        if "net_precipitation" in category:
            cell_type = net_precipitation_type_array
        else:
            cell_type = surface_type_array
        domain_datasets[varname] = conditional_average(
            safe.get_variables(ds, primary_vars), cell_type, category, grid["area"],
        )

    domain_ds = xr.concat(
        [dataset for dataset in domain_datasets.values()], dim="domain"
    ).assign_coords({"domain": (["domain"], [*domain_datasets.keys()])})

    ds = xr.merge([domain_ds, ds.drop(labels=primary_vars)])

    return ds.mean(dim=time_dim, keep_attrs=True)


def insert_column_integrated_vars(
    ds: xr.Dataset, column_integrated_vars: Sequence[str] = PRIMARY_VARS
) -> xr.Dataset:
    """Insert column integrated (<*>) terms,
    really a wrapper around vcm.thermo funcs"""

    for var in column_integrated_vars:
        column_integrated_name = f"column_integrated_{var}"
        if "Q1" in var:
            da = thermo.column_integrated_heating_from_isochoric_transition(
                ds[var], ds[VARNAMES["delp"]]
            )
        elif "Q2" in var:
            da = -thermo.minus_column_integrated_moistening(
                ds[var], ds[VARNAMES["delp"]]
            )
            da = da.assign_attrs(
                {"long_name": "column integrated moistening", "units": "mm/day"}
            )
        else:
            da = mass_integrate(ds[var], ds[VARNAMES["delp"]], dim="z")
        ds = ds.assign({column_integrated_name: da})

    return ds


def _rechunk_time_z(
    ds: xr.Dataset, dim_nchunks: Mapping[str, tuple] = None
) -> xr.Dataset:

    dim_nchunks = dim_nchunks or {"time": 1, "z": ds.sizes["z"]}
    ds = ds.unify_chunks()
    chunks = {dim: ds.sizes[dim] // nchunks for dim, nchunks in dim_nchunks.items()}

    return ds.chunk(chunks)


def conditional_average(
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


def snap_mask_to_type(
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


def units_from_name(var):
    return UNITS.get(var.lower(), "[units unavailable]")


def snap_net_precipitation_to_type(
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

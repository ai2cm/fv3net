from .config import (
    VARNAMES,
    SURFACE_TYPE_ENUMERATION,
    NET_PRECIPITATION_ENUMERATION,
    DOMAINS,
    PRIMARY_VARS,
)
from vcm import thermo, safe
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping, Union, Callable, Any, Tuple

logger = logging.getLogger(__name__)

UNINFORMATIVE_COORDS = ["tile", "z", "y", "x"]
TIME_DIM = "time"
DERIVATION_DIM = "derivation"
PRESSURE_DIM = "pressure"
VERTICAL_DIM = "z"


def reduce_to_diagnostic(
    ds: xr.Dataset,
    grid: xr.Dataset,
    domains: Sequence[str] = DOMAINS,
    primary_vars: Sequence[str] = PRIMARY_VARS,
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds: xarray datasets with relevant variables batched in time
        grid: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
        domains: sequence of area domains over which to produce conditional
            averages; optonal
        primary_vars: sequence of variables for which to compute column integrals
            and composite means
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = ds.drop_vars(names=UNINFORMATIVE_COORDS, errors="ignore")
    ds = _rechunk_time_z(ds)

    grid = grid.drop_vars(names=UNINFORMATIVE_COORDS, errors="ignore")
    surface_type_array = snap_mask_to_type(grid[VARNAMES["surface_type"]])
    if any(["net_precipitation" in category for category in domains]):
        net_precipitation_type_array = snap_mask_to_type(
            ds["net_precipitation"].sel({DERIVATION_DIM: "coarsened_SHiELD"}),
            NET_PRECIPITATION_ENUMERATION,
            np.greater_equal,
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

    return ds.mean(dim=TIME_DIM, keep_attrs=True)


def insert_column_integrated_vars(
    ds: xr.Dataset, column_integrated_vars: Sequence[str] = PRIMARY_VARS
) -> xr.Dataset:
    """Insert column integrated (<*>) terms,
    really a wrapper around vcm.thermo funcs"""

    for var in column_integrated_vars:
        column_integrated_name = f"column_integrated_{var}"
        if "Q1" in var:
            da = thermo.column_integrated_heating(ds[var], ds[VARNAMES["delp"]])
        elif "Q2" in var:
            da = -thermo.minus_column_integrated_moistening(
                ds[var], ds[VARNAMES["delp"]]
            )
            da = da.assign_attrs(
                {"long_name": "column integrated moistening", "units": "mm/day"}
            )
        ds = ds.assign({column_integrated_name: da})

    return ds


def insert_total_apparent_sources(ds: xr.Dataset) -> xr.Dataset:
    """Inserts apparent source (Q) terms as the sum of dQ and pQ, assumed to be present in
    dataset ds
    """
    return ds.assign(
        {
            total_apparent_sources_name: da
            for total_apparent_sources_name, da in zip(
                ("Q1", "Q2"),
                _total_apparent_sources(ds["dQ1"], ds["dQ2"], ds["pQ1"], ds["pQ2"]),
            )
        }
    )


def _total_apparent_sources(
    dQ1: xr.DataArray, dQ2: xr.DataArray, pQ1: xr.DataArray, pQ2: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:

    Q1 = pQ1 + dQ1
    Q2 = pQ2 + dQ2

    return Q1, Q2


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

    all_types = list(np.unique(cell_type_array))

    if category == "global":
        area_masked = area
    elif category in all_types:
        area_masked = area.where(cell_type_array == category)
    else:
        raise ValueError(
            f"surface type {category} not in provided surface type array "
            f"with types {all_types}."
        )

    return weighted_average(ds, area_masked, dims)


def _weighted_average(array, weights, axis=None):

    return np.nansum(array * weights, axis=axis) / np.nansum(weights, axis=axis)


def weighted_average(
    array: Union[xr.Dataset, xr.DataArray],
    weights: xr.DataArray,
    dims: Sequence[str] = ["tile", "y", "x"],
) -> xr.Dataset:
    """Compute a weighted average of an array or dataset
    
    Args:
        array: xr dataarray or dataset of variables to averaged
        weights: xr datarray of grid cell weights for averaging
        dims: dimensions to average over
            
    Returns:
        xr dataarray or dataset of weighted averaged variables
    """
    if dims is not None:
        kwargs = {"axis": tuple(range(-len(dims), 0))}
    else:
        kwargs = {}
    return xr.apply_ufunc(
        _weighted_average,
        array,
        weights,
        input_core_dims=[dims, dims],
        kwargs=kwargs,
        dask="allowed",
    )


def snap_mask_to_type(
    float_mask: xr.DataArray,
    enumeration: Mapping[str, float] = SURFACE_TYPE_ENUMERATION,
    boolean_func: Callable[..., np.ndarray] = np.isclose,
    boolean_func_kwargs: Mapping[str, Any] = None,
) -> xr.DataArray:
    """Convert float surface type array to categorical surface type array
    
    Args:
        float_mask: xr.DataArray of numerical values
        enumeration: mapping of categorical string types to array values
        boolean_func: callable used to map a numerical array to a boolean array
            for a particular value, e.g., np.isclose(arr, value)
        boolean_func_kwargs: dict of args to be passed to boolean_func
            
    Returns:
        types: xr dataarray of categorical str type
    
    """

    boolean_func_kwargs = boolean_func_kwargs or {}

    types = np.full(float_mask.values.shape, np.nan)
    for type_name, type_number in enumeration.items():
        types = np.where(
            boolean_func(float_mask.values, type_number, **boolean_func_kwargs),
            type_name,
            types,
        )

    type_da = xr.DataArray(types, float_mask.coords, float_mask.dims)

    return type_da

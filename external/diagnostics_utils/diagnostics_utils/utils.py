from .config import VARNAMES, SURFACE_TYPE_ENUMERATION
from vcm import thermo, safe
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping, Union

logging.getLogger(__name__)

UNINFORMATIVE_COORDS = ["tile", "z", "y", "x"]
TIME_DIM = "time"


def reduce_to_diagnostic(
    ds_batches: Sequence[xr.Dataset],
    grid: xr.Dataset,
    domains: Sequence[str] = SURFACE_TYPE_ENUMERATION.keys(),
    primary_vars: Sequence[str] = ["dQ1", "pQ1", "dQ2", "pQ2"],
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds_batches: loader sequence of xarray datasets with relevant variables
        grid: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
        domains: sequence of area domains over which to produce conditional
            averages; defaults to ['sea', 'land', 'seaice']
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = xr.concat(ds_batches, dim=TIME_DIM)
    ds = insert_column_integrated_vars(ds, primary_vars)
    ds = _rechunk_time_z(ds)
    ds_time_averaged = ds.mean(dim=TIME_DIM, keep_attrs=True)
    ds_time_averaged = ds_time_averaged.drop_vars(
        names=UNINFORMATIVE_COORDS, errors="ignore"
    )

    grid = grid.drop_vars(names=UNINFORMATIVE_COORDS, errors="ignore")
    surface_type_array = snap_mask_to_type(grid[VARNAMES["surface_type"]])

    conditional_datasets = {}
    for surface_type in domains:
        varname = f"{surface_type}_average"
        conditional_datasets[varname] = conditional_average(
            safe.get_variables(ds_time_averaged, primary_vars),
            surface_type_array,
            surface_type,
            grid["area"],
        )

    domain_ds = xr.concat(
        [dataset for dataset in conditional_datasets.values()], dim="domain"
    ).assign_coords({"domain": (["domain"], [*conditional_datasets.keys()])})

    return xr.merge([domain_ds, ds_time_averaged.drop(labels=primary_vars)])


def insert_column_integrated_vars(
    ds: xr.Dataset, column_integrated_vars: Sequence[str]
) -> xr.Dataset:
    """Insert column integrated (<*>) terms,
    really a wrapper around vcm.thermo funcs"""

    for var in column_integrated_vars:
        column_integrated_name = f"column_integrated_{var}"
        if "Q1" in var:
            da = thermo.column_integrated_heating(ds[var], ds[VARNAMES["delp"]])
        elif "Q2" in var:
            da = thermo.minus_column_integrated_moistening(
                ds[var], ds[VARNAMES["delp"]]
            )
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
    surface_type_array: xr.DataArray,
    surface_type: str,
    area: xr.DataArray,
    dims: Sequence[str] = ["tile", "y", "x"]
) -> xr.Dataset:
    """Average over a conditional type
    
    Args:
        ds: xr dataarray or dataset of variables to averaged conditionally
        surface_type_array: xr datarray of surface type category strings
        surface_type: str of surface type over which to conditionally average
        area: xr datarray of grid cell areas for weighted averaging
            
    Returns:
        xr dataarray or dataset of conditionally averaged variables
    """

    all_types = list(np.unique(surface_type_array))

    if surface_type == "global":
        area_masked = area
    elif surface_type in all_types:
        area_masked = area.where(surface_type_array == surface_type)
    else:
        raise ValueError(
            f"surface type {surface_type} not in provided surface type array "
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
        surface_type: str of surface type over which to conditionally average
        area: xr datarray of grid cell areas for weighted averaging
            
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
    for type_name, type_number in enumeration.items():
        types = np.where(
            np.isclose(float_mask.values, type_number, atol), type_name, types
        )

    types = xr.DataArray(types, float_mask.coords, float_mask.dims)

    return types

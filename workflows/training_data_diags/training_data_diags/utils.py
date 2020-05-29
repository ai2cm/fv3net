from .config import VARNAMES, SURFACE_TYPE_ENUMERATION
from fv3net.regression import loaders
from vcm import mass_integrate, thermo, safe
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping

logging.getLogger(__name__)


def reduce_to_diagnostic(
    ds_batches: Sequence[xr.Dataset],
    static_variables: xr.Dataset,
    domains: Sequence[str]=SURFACE_TYPE_ENUMERATION.keys(),
    primary_vars: Sequence[str]=['dQ1', 'pQ1', 'dQ2', 'pQ2']
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds_batches: loader sequence of xarray datasets with relevant variables
        static_variables: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
        domains: sequence of area domains over which to produce conditional
            averages; defaults to ['sea', 'land', 'seaice']
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """
    
    ds_list = []
    for ds in ds_batches:
        ds_list.append(_insert_column_integrated_vars(ds, primary_vars))

    ds_time_averaged = _time_average(ds_list).drop(labels=VARNAMES['delp_var'])

    conditional_datasets = {}
    surface_type_array = snap_mask_to_type(static_variables[VARNAMES['surface_type_var']])
    area = static_variables["area"]
    for surface_type in domains:
        varname = f"ds_{surface_type}_average"
        conditional_datasets[varname] = _conditional_average(
            safe.get_variables(ds_time_averaged, primary_vars),
            surface_type_array,
            surface_type,
            area
        )

    domain_ds = xr.concat(
        [dataset for dataset in conditional_datasets.values()], dim="domain"
    ).assign_coords({"domain": (['domain'], [*conditional_datasets.keys()])})
    
    return xr.merge([domain_ds, ds_time_averaged.drop(labels=primary_vars)])


def _insert_column_integrated_vars(
    ds: xr.Dataset,
    column_integrated_vars: Sequence[str]
) -> xr.Dataset:
    '''Insert column integreated (<*>) terms,
    really a wrapper around vcm.thermo funcs'''
    
    
    for var in column_integrated_vars:
        column_integrated_name = f"column_integrated_{var}"
        if 'Q1' in var:
            da = thermo.column_integrated_heating(
                ds[var],
                ds['pressure_thickness_of_atmospheric_layer']
            )
        elif 'Q2' in var:
            da = thermo.minus_column_integrated_moistening(
                ds[var],
                ds['pressure_thickness_of_atmospheric_layer']
            )
        ds = ds.assign({column_integrated_name: da})
    
    return ds


def _time_average(batches: Sequence[xr.Dataset], time_dim="time") -> xr.Dataset:
    """Average over time dimension"""

    ds = xr.concat(batches, dim=time_dim)

    return ds.mean(dim=time_dim, keep_attrs=True)


def _conditional_average(
    ds: xr.Dataset,
    surface_type_array: xr.DataArray,
    surface_type: str,
    area: xr.DataArray
) -> xr.Dataset:
    """Average over a conditional type"""
    
    all_types = list(np.unique(surface_type_array))
    
    if surface_type == 'global': 
        area_masked = area
    elif surface_type in all_types:
        area_masked = area.where(surface_type_array == surface_type)
    else:
        raise ValueError(
            f'surfae type {surface_type} not in provided surface type array '
            f'with types {all_types}.'
        )

    return weighted_average(ds, area_masked)


def _weighted_average(array, weights, axis=None):

    return np.nansum(array * weights, axis=axis)/np.nansum(weights, axis=axis)


def weighted_average(array: xr.Dataset, weights: xr.DataArray, dims: Sequence[str] = ["tile", "y", "x"]) -> xr.Dataset:
    if dims is not None:
        kwargs = {'axis': tuple(range(-len(dims), 0))}
    else:
        kwargs = {}
    return xr.apply_ufunc(
        _weighted_average,
        array,
        weights,
        input_core_dims=[dims, dims],
        kwargs=kwargs,
        dask='allowed'
    )


def snap_mask_to_type(
    float_mask: xr.DataArray,
    enumeration: Mapping = SURFACE_TYPE_ENUMERATION,
    atol: float = 1e-7,
) -> xr.DataArray:
    """Convert float surface type array to categorical surface type array"""

    types = np.empty_like(float_mask.values)
    for type_name, type_number in enumeration.items():
        types = np.where(np.isclose(float_mask.values, type_number, atol), type_name, types)
    
    types = xr.DataArray(types, float_mask.coords, float_mask.dims)
    
    return types

from fv3net.regression import loaders
from vcm import mass_integrate
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping

logging.getLogger(__name__)

SURFACE_TYPE_ENUMERATION = {"sea": 0, "land": 1, "seaice": 2}


def reduce_to_diagnostic(
    ds_batches: Sequence[xr.Dataset], static_variables: xr.Dataset
) -> xr.Dataset:
    """Reduce a sequence of batches to a diagnostic dataset
    
    Args:
        ds_batches: loader sequence of xarray datasets with relevant variables
        static_variables: xarray dataset containing grid variables
        (latb, lonb, lat, lon, area, land_sea_mask)
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    area = static_variables["area"]
    surface_type_array = snap_mask_to_type(static_variables["land_sea_mask"])

    ds_time_averaged = time_average(ds_batches)

    conditional_datasets = {}
    for surface_type in SURFACE_TYPE_ENUMERATION:
        varname = f"ds_{surface_type}_average"
        conditional_datasets[varname] = conditional_time_average(
            ds_time_averaged, surface_type_array, surface_type, area
        )

    diagnostic_ds = xr.concat(
        [dataset for dataset in conditional_datasets.values()], dim="domain"
    ).assign_coords({"domain": conditional_datasets.keys()})

    return diagnostic_ds


def time_average(batches: Sequence[xr.Dataset], time_dim="time") -> xr.Dataset:
    """Average over time dimension"""

    ds = xr.concat(batches, dim=time_dim)

    return ds.mean(dim=time_dim, keep_attrs=True)


def conditional_time_average(
    ds: xr.Dataset,
    surface_type_array: xr.DataArray,
    surface_type: str,
    area: xr.DataArray,
) -> xr.Dataset:
    """Average over a conditional type"""

    area_masked = area.where(surface_type_array == surface_type)
    ds_time_average = xr.Dataset()
    for var in ds:
        ds_time_average.assign({var: _weighted_mean(ds[var], area_masked)})

    return ds_time_average


def _weighted_average(
    array: xr.DataArray, weights: xr.DataArray, dims: str = ["tile", "y", "x"]
) -> xr.DataArray:

    return (array * weights).mean(dim=dims).assign_attrs(array.attrs)


def snap_mask_to_type(
    float_mask: xr.DataArray,
    enumeration: Mapping = SURFACE_TYPE_ENUMERATION,
    atol: float = 1e-7,
) -> xr.DataArray:
    """Convert float surface type array to categorical surface type array"""

    types = np.empty_like(float_mask.values)
    for type_number, type_name in enumeration.items():
        types.where(np.isclose(float_mask.values, type_number, atol), type_name)

    return types

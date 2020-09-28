from .config import (
    VARNAMES,
    SURFACE_TYPE_ENUMERATION,
    DOMAINS,
    PRIMARY_VARS,
)
from vcm import thermo, safe
import xarray as xr
import numpy as np
import logging
from typing import Sequence, Mapping, Union, Tuple

logger = logging.getLogger(__name__)


def reduce_to_diagnostic(
    ds: xr.Dataset,
    grid: xr.Dataset,
    domains: Sequence[str] = DOMAINS,
    primary_vars: Sequence[str] = PRIMARY_VARS,
    net_precipitation: xr.DataArray = None,
    time_dim: str = "time",
    derivation_dim: str = "derivation",
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
        derivation_dim: name of the dataset derivation dimension containing coords
            such as 'target', 'predict', etc.; optional, defaults to 'derivation'
        uninformative_coords: sequence of names of uninformative (i.e.,
            range(len(dim))), coordinates to be dropped
            
    Returns:
        diagnostic_ds: xarray dataset of reduced diagnostic variables
    """

    ds = ds.drop_vars(names=uninformative_coords, errors="ignore")
    ds = _rechunk_time_z(ds)

    grid = grid.drop_vars(names=uninformative_coords, errors="ignore")
    surface_type_array = snap_mask_to_type(grid[VARNAMES["surface_type"]])
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


def insert_net_terms_as_Qs(
    ds: xr.Dataset,
    var_mapping: Mapping = None,
    derivation_dim: str = "derivation",
    shield_coord: str = "coarsened_SHiELD",
    derivations_keep: Sequence[str] = ("target", "predict"),
) -> xr.Dataset:
    """Insert the SHiELD net_* variables as the column_integrated_Q* variables
        for coordinate 'coarsened_SHiELD', also drop the net_* variables and the
        'coarse_FV3GFS' coordinate; this is useful in the offline_ML_diags routine
        because eliminates an unnecessary coordinate and includes SHiELD variables
        in the calculated diagnostics and metrics
        
    Args:
        ds: xr dataset to from which to compute diagnostics
        var_mapping: dict which maps SHiELD net_* var names to
            column_integrated_Q* var names; optional
        derivation_dim: name of derivation dim; optional, defaults to 'derivation'
        shield_coord: name of SHiELD coordinate in derivation dim; optional
        derivations_keep: sequence of derivation coords to keep in output dataset
            
    Returns:
        xr dataset of renamed and rearranged variables
    """
    var_mapping = var_mapping or {
        "net_heating": "column_integrated_Q1",
        "net_precipitation": "column_integrated_Q2",
    }

    ds_new = ds.sel({derivation_dim: list(derivations_keep)}).drop_vars(
        names=var_mapping.keys(), errors="ignore"
    )

    shield_data = {}
    for var_source_name, var_target_name in var_mapping.items():
        if var_source_name in ds.data_vars:
            if "Q1" in var_target_name:
                shield_data[var_target_name] = ds[var_source_name].sel(
                    {derivation_dim: [shield_coord]}
                )
            elif "Q2" in var_target_name:
                shield_data[var_target_name] = -ds[var_source_name].sel(
                    {derivation_dim: [shield_coord]}
                )

    return ds_new.merge(shield_data)


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


def _units_from_Q_name(var):
    if "r2" in var.lower():
        return ""
    if "q1" in var.lower():
        if "column_integrated" in var:
            return "[W/m^2]"
        else:
            return "[K/s]"
    elif "q2" in var.lower():
        if "column_integrated" in var:
            return "[mm/day]"
        else:
            return "[kg/kg/s]"
    else:
        return None


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

import functools
import intake
import numpy as np
from toolz import compose
from typing import Tuple, Sequence, Mapping
import xarray as xr
import vcm

from ..constants import TIME_NAME

EDGE_TO_CENTER_DIMS = {
    "x_interface": "x",
    "y_interface": "y"
}


def nonderived_variable_names(
        variable_names,
        cos_z_var: str = "cos_zenith_angle",
        latlon_wind_tendency_vars: Tuple[str] = None,
        xy_wind_tendency_vars: Tuple[str] = None):
    latlon_wind_tendency_vars = latlon_wind_tendency_vars or ["dQu", "dQv"]
    xy_wind_tendency_vars = xy_wind_tendency_vars or ["dQx", "dQy"]
    derived_variables = latlon_wind_tendency_vars + [cos_z_var]
    nonderived_variables = [
        var for var in variable_names if var not in derived_variables]
    # need to load x/y wind tendencies to derive lat/lon components
    if any(var in variable_names for var in latlon_wind_tendency_vars):
        nonderived_variables += xy_wind_tendency_vars
    return nonderived_variables


def insert_derived_variables(
    variables: Sequence[str],
    cos_z_var: str = "cos_zenith_angle",
    xy_wind_tendency_vars: Sequence[str] = None,
    latlon_wind_tendency_vars: Sequence[str] = None,
    catalog_path: str = "catalog.yml",
    res: str = "c48",
    edge_to_center_dims: Mapping = None,
):
    """Checks if any of the derived variables are requested in the
    model configuration, and for each derived variable adds partial function
    to inserts them into the final dataset.

    Args:
        cos_z_var: Name for the cosine zenith angle derived variable.
            Defaults to "cos_zenith_angle".
        xy_wind_tendency_vars: Names of x and y wind tendencies (in that order)
            in the local grid basis. Defaults to ["dQx", "dQy"].
        latlon_wind_tendency_vars: Names to assign to rotated wind tendencies
            in the lat/lon basis. Defaults to ["dQu", "dQv"].
        catalog_path: Path to catalog. Defaults to "catalog.yml"
            (assumes running from top level of fv3net dir).

    Returns:
        Composed partial function that inserts the derived variables into the
        batch dataset.
    """
    xy_wind_tendency_vars = xy_wind_tendency_vars or ["dQx", "dQy"]
    latlon_wind_tendency_vars = latlon_wind_tendency_vars or ["dQu", "dQv"]
    edge_to_center_dims = edge_to_center_dims or EDGE_TO_CENTER_DIMS
    derived_var_partial_funcs = []
    
    if cos_z_var in variables:
        grid = _load_grid(res, catalog_path)
        derived_var_partial_funcs.append(
            functools.partial(_insert_cos_z, grid, cos_z_var))
    if len(set(variables) & set(latlon_wind_tendency_vars)) > 0:
        wind_rotation_matrix = _load_wind_rotation_matrix(res, catalog_path)
        derived_var_partial_funcs.append(
            functools.partial(
                _insert_latlon_wind_tendencies,
                wind_rotation_matrix,
                xy_wind_tendency_vars,
                latlon_wind_tendency_vars)
        )
        derived_var_partial_funcs.append(
            functools.partial(
                _center_d_grid_winds,
                xy_wind_tendency_vars,
                edge_to_center_dims)
        )
    return compose(*derived_var_partial_funcs)


def _center_d_grid_winds(
    xy_wind_tendency_vars: Sequence[str],
    edge_to_center_dims: Mapping,
    ds: xr.Dataset
):
    edge_to_center_dims = edge_to_center_dims or EDGE_TO_CENTER_DIMS
    for edge_wind in xy_wind_tendency_vars:
        ds[edge_wind] = vcm.cubedsphere.shift_edge_var_to_center(
            ds[edge_wind], edge_to_center_dims)
    return ds


def _insert_latlon_wind_tendencies(
    wind_rotation_matrix: xr.Dataset,
    xy_wind_tendency_vars: Sequence[str],
    latlon_wind_tendency_vars: Sequence[str],
    ds: xr.Dataset
):
    x_tendency, y_tendency = xy_wind_tendency_vars
    lat_tendency, lon_tendency = latlon_wind_tendency_vars

    ds[lat_tendency] = (
        wind_rotation_matrix["eastward_wind_u_coeff"] * ds[x_tendency]
        + wind_rotation_matrix["eastward_wind_v_coeff"] * ds[y_tendency]
    )
    ds[lon_tendency] = (
        wind_rotation_matrix["northward_wind_u_coeff"] * ds[x_tendency]
        + wind_rotation_matrix["northward_wind_v_coeff"] * ds[y_tendency]
    )
    return ds


def _load_grid(res="c48", catalog_path="catalog.yml"):
    cat = intake.open_catalog(catalog_path)
    grid = cat[f"grid/{res}"].to_dask()
    land_sea_mask = cat[f"landseamask/{res}"].to_dask()
    grid = grid.assign({"land_sea_mask": land_sea_mask["land_sea_mask"]})
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])
    return grid


def _load_wind_rotation_matrix(res="c48", catalog_path="catalog.yml"):
    cat = intake.open_catalog(catalog_path)
    return cat[f"wind_rotation/{res}"].to_dask()


def _insert_cos_z(
    grid: xr.Dataset, cos_z_var: str, ds: xr.Dataset
) -> xr.Dataset:
    times_exploded = np.array(
        [
            np.full(grid["lon"].shape, vcm.cast_to_datetime(t))
            for t in ds[TIME_NAME].values
        ]
    )
    cos_z = vcm.cos_zenith_angle(times_exploded, grid["lon"], grid["lat"])
    return ds.assign({cos_z_var: ((TIME_NAME,) + grid["lon"].dims, cos_z)})



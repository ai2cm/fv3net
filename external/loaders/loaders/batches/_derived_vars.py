import functools
import intake
import numpy as np
from toolz import compose
from typing import Sequence
import xarray as xr
import vcm

from ..constants import TIME_NAME

EDGE_TO_CENTER_DIMS = {"x_interface": "x", "y_interface": "y"}

COS_Z = "cos_zenith_angle"
EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
X_Y_WIND_TENDENCIES = ["dQxwind", "dQywind"]


def nonderived_variable_names(requested: Sequence[str], available: Sequence[str]):
    derived = [var for var in requested if var not in available]
    nonderived = [var for var in requested if var in available]
    # if E/N winds not in underlying data, need to load x/y wind
    # tendencies to derive them
    if any(var in derived for var in EAST_NORTH_WIND_TENDENCIES):
        nonderived += X_Y_WIND_TENDENCIES
    return nonderived


def insert_derived_variables(
    variables: Sequence[str], catalog_path: str = "catalog.yml", res: str = "c48",
):
    """Checks if any of the derived variables are requested in the
    model configuration, and for each derived variable adds partial function
    to inserts them into the final dataset.

    Args:
        catalog_path: Path to catalog. Defaults to "catalog.yml"
            (assumes running from top level of fv3net dir).

    Returns:
        Composed partial function that inserts the derived variables into the
        batch dataset.
    """
    derived_var_partial_funcs = []

    if COS_Z in variables:
        grid = _load_grid(res, catalog_path)
        derived_var_partial_funcs.append(functools.partial(_insert_cos_z, grid))
    if any(var in variables for var in EAST_NORTH_WIND_TENDENCIES):
        wind_rotation_matrix = _load_wind_rotation_matrix(res, catalog_path)
        derived_var_partial_funcs.append(
            functools.partial(
                vcm.cubedsphere.insert_eastnorth_wind_tendencies, wind_rotation_matrix,
            )
        )
    return compose(*derived_var_partial_funcs)


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


def _insert_cos_z(grid: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
    times_exploded = np.array(
        [
            np.full(grid["lon"].shape, vcm.cast_to_datetime(t))
            for t in ds[TIME_NAME].values
        ]
    )
    cos_z = vcm.cos_zenith_angle(times_exploded, grid["lon"], grid["lat"])
    return ds.assign({COS_Z: ((TIME_NAME,) + grid["lon"].dims, cos_z)})

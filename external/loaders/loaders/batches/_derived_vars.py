import functools
from toolz import compose
import .._utils as utils

"""
if cos_z_var in variable_names:
    grid = load_grid()
    insert_cos_z = functools.partial(add_cosine_zenith_angle, grid, cos_z_var)
    batch_func = compose(transform, insert_cos_z, load_batch)
else:
    batch_func = compose(transform, load_batch)
"""


def insert_derived_variables(
    variables: Sequence[str],
    cos_z_var: str = "cos_zenith_angle",
    xy_wind_tendency_vars: Sequence[str] = None,
    latlon_wind_tendency_vars: Sequence[str] = None,
    catalog_path: str = "catalog.yml",
    res: str = "c48"
):
    """Checks if any of the derived variables are requested in the
    model configuration, and for each derived variable adds partial function
    to inserts them into the final dataset.

    Args:
        cos_z_var: Name for the cosine zenith angle derived variable.
            Defaults to "cos_zenith_angle".
        xy_wind_tendency_vars: Names of wind tendencies in the x/y basis.
            Defaults to ["dQu", "dQv"].
        latlon_wind_tendency_vars: Names to assign to rotated wind tendencies
            in the lat/lon basis. Defaults to overwriting ["dQu", "dQv"].
        catalog_path: Path to catalog. Defaults to "catalog.yml"
            (assumes running from top level of fv3net dir).

    Returns:
        Composed partial function that inserts the derived variables into the
        batch dataset.
    """
    xy_wind_tendency_vars = xy_wind_tendency_vars or ["dQu", "dQv"]
    latlon_wind_tendency_vars = latlon_wind_tendency_vars or ["dQu", "dQv"]

    derived_var_partial_funcs = []
    
    if cos_z_var in variables:
        grid = _load_grid(res, catalog_path)
        derived_var_partial_funcs.append(
            functools.partial(_add_cosine_zenith_angle, grid, cos_z_var))
    if len(set(variables) & set(wind_tendency_vars)) > 0:
        wind_rotation_matrix = _load_wind_rotation_matrix(res, catalog_path)
        derived_var_partial_funcs.append(
            functools.partial(
                _rotate_wind_tendencies,
                wind_rotation_matrix,
                xy_wind_tendency_vars,
                latlon_wind_tendency_vars)
        )
    return compose(*derived_var_partial_funcs)


def _rotate_wind_tendencies(
    wind_rotation_matrix: xr.Dataset,
    xy_wind_tendency_vars: Sequence[str],
    latlon_wind_tendency_vars: Sequence[str],
    ds: xr.Dataset
):
    


def _load_grid(res="c48", catalog_path="catalog.yml"):
    cat = intake.open_catalog(catalog_path)
    grid = cat[f"grid/{res}"].to_dask()
    land_sea_mask = cat[f"landseamask/{res}"].to_dask()
    grid = grid.assign({"land_sea_mask": land_sea_mask["land_sea_mask"]})
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])
    return grid


def _load_wind_rotation_matrix(res="c48", catalog_path="catalog.yml"):
    cat = intake.open_catalog(catalog_path)
    return cat[f"wind_rotation_matrix/{res}"].to_dask()


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




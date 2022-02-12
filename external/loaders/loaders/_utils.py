import numpy as np
from typing import Any, Hashable, Mapping, Tuple, Sequence
from toolz.functoolz import curry
import xarray as xr
import vcm
from vcm import safe, net_heating, net_precipitation, DerivedMapping
from vcm.convenience import round_time

from .constants import TIME_NAME
from vcm.catalog import catalog


# note this is intentionally a different value than fv3fit's SAMPLE_DIM_NAME
# as we should avoid mixing loaders preprocessing routines with fv3fit
# preprocessing routines - migrate the routines here instead

SAMPLE_DIM_NAME = "_fv3net_sample"
CLOUDS_OFF_TEMP_TENDENCIES = [
    "tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_turbulence",
    "tendency_of_air_temperature_due_to_dissipation_of_gravity_waves",
]
CLOUDS_OFF_SPHUM_TENDENCIES = ["tendency_of_specific_humidity_due_to_turbulence"]
EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
X_Y_WIND_TENDENCIES = ["dQxwind", "dQywind"]
WIND_ROTATION_COEFFICIENTS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
]

Time = str
Tile = int
K = Tuple[Time, Tile]


def nonderived_variables(requested: Sequence[Hashable], available: Sequence[Hashable]):
    derived = [var for var in requested if var not in available]
    nonderived = [var for var in requested if var in available]
    # if E/N winds not in underlying data, need to load x/y wind
    # tendencies to derive them
    # TODO move to derived_mapping?
    if any(var in derived for var in EAST_NORTH_WIND_TENDENCIES):
        nonderived += X_Y_WIND_TENDENCIES
    if any(var in derived for var in ["eastward_wind", "northward_wind"]):
        nonderived += ["x_wind", "y_wind"]
    return nonderived


def stack(unstacked_dims: Sequence[str], ds: xr.Dataset) -> xr.Dataset:
    """
    Stack dimensions into a sample dimension, retaining other dimensions
    in alphabetical order.

    Args:
        unstacked_dims: dimensions to keep, other dimensions will be
            collapsed into the sample dimension
        ds: dataset to stack
    """
    stack_dims = [dim for dim in ds.dims if dim not in unstacked_dims]
    unstacked_dims = [dim for dim in ds.dims if dim in unstacked_dims]
    unstacked_dims.sort()  # needed to always get [x, y, z] dimensions
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=list(unstacked_dims) + ["time", "dataset"],
    )
    return ds_stacked.transpose(SAMPLE_DIM_NAME, *unstacked_dims)


def shuffle(ds: xr.Dataset) -> xr.Dataset:
    shuffled_indices = np.arange(len(ds[SAMPLE_DIM_NAME]), dtype=int)
    np.random.shuffle(shuffled_indices)
    return ds.isel({SAMPLE_DIM_NAME: shuffled_indices})


def dropna(ds: xr.Dataset) -> xr.Dataset:
    """
    Check for an empty variables along a dimension in a dataset
    """
    ds = ds.dropna(dim=SAMPLE_DIM_NAME)
    if len(ds[SAMPLE_DIM_NAME]) == 0:
        raise ValueError("Check for NaN fields in the training data.")
    return ds


@curry
def add_derived_data(variables: Sequence[str], ds: xr.Dataset) -> xr.Dataset:
    """
    Overlay the DerivedMapping and grab a dataset of specified variables

    Args:
        variables: All variables (derived and non-derived) to include in the
            dataset.
    """
    derived_mapping = DerivedMapping(ds)
    return derived_mapping.dataset(variables)


@curry
def add_grid_info(res: str, ds: xr.Dataset) -> xr.Dataset:
    """
    Add lat, lon, land-type mask information to the dataset

    Args:
        res: grid resolution, format as f'c{number cells in tile}'
    """
    grid = _load_grid(res)
    # Prioritize dataset's land_sea_mask if it differs from grid
    return xr.merge([ds, grid], compat="override")


@curry
def add_wind_rotation_info(res: str, ds: xr.Dataset) -> xr.Dataset:
    """
    Add wind rotation information to the dataset

    Args:
        res: grid resolution, format as f'c{number cells in tile}'
    """

    rotation = _load_wind_rotation_matrix(res).drop("tile")
    common_coords = {"x": ds["x"].values, "y": ds["y"].values}
    rotation = rotation.assign_coords(common_coords)
    return ds.merge(rotation, compat="override")


def _load_grid(res: str) -> xr.Dataset:
    grid = catalog[f"grid/{res}"].to_dask()
    land_sea_mask = catalog[f"landseamask/{res}"].to_dask()
    grid = grid.assign({"land_sea_mask": land_sea_mask["land_sea_mask"]})
    # drop the tiles so that this is compatible with other indexing conventions
    return safe.get_variables(grid, ["lat", "lon", "land_sea_mask"]).drop("tile")


def _load_wind_rotation_matrix(res: str) -> xr.Dataset:
    rotation = catalog[f"wind_rotation/{res}"].to_dask()
    return safe.get_variables(rotation, WIND_ROTATION_COEFFICIENTS)


def get_sample_dataset(mapper):
    sample_key = list(mapper.keys())[0]
    return mapper[sample_key]


def standardize_zarr_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """ Casts a datetime coord to to python datetime and rounds to
    nearest even second (because cftime coords have small rounding
    errors that makes it hard to other datasets join on time)

    Args:
        ds (xr.Dataset): time coordinate is datetime-like object

    Returns:
        xr.Dataset with standardized time coordinates
    """
    # Vectorize doesn't work on type-dispatched function overloading
    times = np.array(list(map(vcm.cast_to_datetime, ds[TIME_NAME].values)))
    times = round_time(times)
    ds = ds.assign_coords({TIME_NAME: times})
    return ds


def net_heating_from_physics(ds: xr.Dataset) -> xr.DataArray:

    fluxes = (
        ds["total_sky_downward_longwave_flux_at_surface"],
        ds["total_sky_downward_shortwave_flux_at_surface"],
        ds["total_sky_upward_longwave_flux_at_surface"],
        ds["total_sky_upward_longwave_flux_at_top_of_atmosphere"],
        ds["total_sky_upward_shortwave_flux_at_surface"],
        ds["total_sky_upward_shortwave_flux_at_top_of_atmosphere"],
        ds["total_sky_downward_shortwave_flux_at_top_of_atmosphere"],
        ds["sensible_heat_flux"],
        ds["surface_precipitation_rate"],
    )
    return net_heating(*fluxes)


def net_precipitation_from_physics(ds: xr.Dataset) -> xr.DataArray:

    fluxes = (
        ds["latent_heat_flux"],
        ds["surface_precipitation_rate"],
    )
    return net_precipitation(*fluxes)


def assign_net_physics_terms(ds: xr.Dataset) -> xr.Dataset:
    net_terms: Mapping[Hashable, Any] = {
        "net_heating": net_heating_from_physics(ds),
        "net_precipitation": net_precipitation_from_physics(ds),
    }
    return ds.assign(net_terms)

import numpy as np
from numpy.random import RandomState
from typing import Tuple
import xarray as xr
import vcm
from vcm import safe, net_heating, net_precipitation
from vcm.convenience import round_time

from .constants import SAMPLE_DIM_NAME, TIME_NAME


CLOUDS_OFF_TEMP_TENDENCIES = [
    "tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_turbulence",
    "tendency_of_air_temperature_due_to_dissipation_of_gravity_waves",
]
CLOUDS_OFF_SPHUM_TENDENCIES = ["tendency_of_specific_humidity_due_to_turbulence"]
Z_DIM_NAMES = ["z", "pfull"]
DATASET_DIM_NAME = "dataset"

Time = str
Tile = int
K = Tuple[Time, Tile]


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
    times = np.vectorize(round_time)(times)
    ds = ds.assign_coords({TIME_NAME: times})
    return ds


def stack_dropnan_shuffle(random_state: RandomState, ds: xr.Dataset,) -> xr.Dataset:
    ds = ds.load()
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + [TIME_NAME, DATASET_DIM_NAME],
    )
    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM_NAME)
    if len(ds_no_nan[SAMPLE_DIM_NAME]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )
    ds_no_nan = ds_no_nan.transpose()
    result = shuffled(ds_no_nan, SAMPLE_DIM_NAME, random_state)
    if DATASET_DIM_NAME in ds.dims:
        # In the multi-dataset case, preserve the same number of samples per
        # batch as the single dataset case.
        return result.thin({SAMPLE_DIM_NAME: ds.sizes[DATASET_DIM_NAME]})
    else:
        return result


def shuffled(
    dataset: xr.Dataset, dim: str, random: np.random.RandomState
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dataset: input data to be shuffled
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


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


def assign_net_physics_terms(ds: xr.Dataset) -> xr.DataArray:
    net_terms = {
        "net_heating": net_heating_from_physics(ds),
        "net_precipitation": net_precipitation_from_physics(ds),
    }
    return ds.assign(net_terms)


def compute_clouds_off_pQ1(ds: xr.Dataset) -> xr.DataArray:
    """Compute the clouds off tendency of temperature.

    The input Dataset must contain the physics tendency component
    diagnostics as output by the fortran model.

    Args:
        ds: input Dataset
    Returns:
        A DataArray with the clouds off temperature tendency
    """
    return sum([ds[variable] for variable in CLOUDS_OFF_TEMP_TENDENCIES])


def compute_clouds_off_pQ2(ds: xr.Dataset) -> xr.DataArray:
    """Compute the clouds off tendency of specific humidity.
    
    The input Dataset must contain the physics tendency component
    diagnostics as output by the fortran model.

    Args:
        ds: input Dataset
    Returns:
        A DataArray with the clouds off specific humidity tendency
    """
    return sum([ds[variable] for variable in CLOUDS_OFF_SPHUM_TENDENCIES])

from typing import Callable

import cftime
import xarray as xr

import pace.util
from runtime.types import State
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state


def scatter_within_tile(
    time: cftime.DatetimeJulian,
    time_lookup_function: Callable[[cftime.DatetimeJulian], State],
    communicator: pace.util.CubedSphereCommunicator,
) -> xr.Dataset:
    """Scatter data for a timestamp from each tile's master rank to its subranks.

    Args:
        time: time of the data
        time_lookup_function: a function that takes a time and returns a state dict
            containing data arrays to be scattered
        communicator: model cubed sphere communicator

    Returns:
        Dataset of scattered data arrays
    """
    if communicator.tile.rank == 0:
        state: State = time_lookup_function(time)
    else:
        state = {}

    tile = communicator.partitioner.tile_index(communicator.rank)
    if communicator.tile.rank == 0:
        ds = xr.Dataset(state).isel(tile=tile)
        scattered_state = communicator.tile.scatter_state(dataset_to_quantity_state(ds))
    else:
        scattered_state = communicator.tile.scatter_state()

    return quantity_state_to_dataset(scattered_state)

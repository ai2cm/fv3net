from typing import Callable

import cftime
import logging
import xarray as xr

import pace.util
from runtime.types import State
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state


logger = logging.getLogger(__name__)


def scatter_within_tile_for_prescriber(
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
        tile = communicator.partitioner.tile_index(communicator.rank)
        ds = xr.Dataset(state).isel(tile=tile)
    else:
        ds = xr.Dataset({})

    return scatter_within_tile(communicator, ds)


def scatter_within_tile(
    communicator: pace.util.CubedSphereCommunicator, state: State,
) -> xr.Dataset:
    """Scatter data from each tile's master rank to its subranks.

    Args:
        communicator: model cubed sphere communicator

    Returns:
        Dataset of scattered data arrays
    """
    state = xr.Dataset(state)
    if communicator.tile.rank == 0:
        scattered_state = communicator.tile.scatter_state(
            dataset_to_quantity_state(state)
        )
    else:
        scattered_state = communicator.tile.scatter_state()

    return quantity_state_to_dataset(scattered_state)


def gather_from_subtiles(
    communicator: pace.util.CubedSphereCommunicator, state: State,
) -> xr.Dataset:
    """Gather data from each sub rank onto the root tile.

    Args:
        communicator: model cubed sphere communicator

    Returns:
        Dataset of gathered data arrays
    """
    state = xr.Dataset(state)
    gathered_state = communicator.tile.gather_state(dataset_to_quantity_state(state))

    if communicator.tile.rank == 0:
        return quantity_state_to_dataset(gathered_state)
    else:
        return None

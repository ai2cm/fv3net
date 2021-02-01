import logging
import os
import fsspec
import xarray as xr
from typing import Sequence

from ._xarray import XarrayMapper

logger = logging.getLogger(__name__)


def open_phys_emu_training(
    url: str, init_times: Sequence[str], consolidated: bool = True,
):
    """
    Load training data for the all-physics emulation.  Combines specified initialization
    time runs together into a single dataset.
    
    Args:
        url (str): path to a nudge-to-obs output directory, remote or local
        init_times: list of initialization times to combine into a single dataset
        consolidated (bool): whether zarrs to open have consolidated metadata
        
    Returns:
        mapper to dataset containing nudging physics tendencies
            and model state data
        
    """
    logger.info(f"Loading emulation training from: {url}")
    logger.info(f"From times: {init_times}")

    # state_after_timestep currently constists of state after "step_dynamics"
    dataset_names = ["physics_tendencies.zarr", "state_after_timestep.zarr"]
    loaded_time_ds = []
    for init_t in init_times:
        loaded = []
        for src in dataset_names:
            mapper = fsspec.get_mapper(os.path.join(url, init_t, src))
            ds = xr.open_zarr(mapper, consolidated=consolidated)
            loaded.append(ds)
        loaded_time_ds.append(xr.merge(loaded))

    # TODO: Doesn't seem to care about overlapping times with different data
    full_ds = xr.merge(loaded_time_ds, dim="time")

    return XarrayMapper(full_ds)

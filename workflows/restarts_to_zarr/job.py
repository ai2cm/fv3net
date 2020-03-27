from typing import Sequence, Hashable
from collections import MutableMapping
import zarr
import xarray as xr
from distributed import Client
import numpy as np
from toolz import curry
import fv3net



def load_timestep(timestep_url):
    pass


@curry
def get_timestep(timesteps, index):
    return load_timestep(timesteps[index])


def write_to_group(group: zarr.Group, ds: xr.Dataset):
    for variable in group:
        group[variable] = np.asarray(ds[variable])


def insert_timestep(output: fv3net.ZarrMapping, time: Hashable):
    output[time] = get_timestep(time)


def restarts_to_zarr(client: Client, store: zarr.ABSStore, timestep_urls: Sequence[str], schema: xr.Dataset):
    group = zarr.open_group(store)
    output_m = fv3net.ZarrMapping(group, schema, timestep_urls, dim='time')
    for i, time_url in enumerate(timestep_urls):
        client.submit(insert_timestep, group, i, get_timestep, pure=False)

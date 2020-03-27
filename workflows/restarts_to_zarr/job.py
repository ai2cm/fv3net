from typing import Sequence
import zarr
import xarray as xr
from distributed import Client
import numpy as np
from toolz import curry


def load_timestep(timestep_url):
    pass


@curry
def get_timestep(timesteps, index):
    return load_timestep(timesteps[index])


def write_to_group(group: zarr.Group, ds: xr.Dataset):
    for variable in group:
        group[variable] = np.asarray(ds[variable])


def insert_timestep(group, index, get):
    ds = get(index)
    write_to_group(group, ds)


def create_zarr(store: zarr.ABSStore, schema: xr.Dataset, dim: str, values: Sequence) -> zarr.Group:
    group = zarr.open_group(store)
    # TODO refactor zarr init from the one-step-job


    return group


def restarts_to_zarr(client: Client, store: zarr.ABSStore, timestep_urls: Sequence[str], schema: xr.Dataset):
    group = create_zarr(store, schema, timestep_urls, new_dim=['time'])
    for i, time_url in enumerate(timestep_urls):
        client.submit(insert_timestep, group, i, get_timestep, pure=False)

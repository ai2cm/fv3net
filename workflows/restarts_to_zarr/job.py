from typing import Sequence, Hashable
import zarr
import xarray as xr
from distributed import Client
from toolz import curry
import fv3net


@curry
def get_timestep(timesteps, index):
    return load_timestep(timesteps[index])


def insert_timestep(output: fv3net.ZarrMapping, time: Hashable):
    output[time] = get_timestep(time)


def restarts_to_zarr(client: Client, store: zarr.ABSStore, timestep_urls: Sequence[str], schema: xr.Dataset):
    group = zarr.open_group(store)
    output_m = fv3net.ZarrMapping(group, schema, timestep_urls, dim='time')
    for i, time_url in enumerate(timestep_urls):
        client.submit(insert_timestep, output_m, time_url, pure=False)

import logging
import os
from itertools import product
from typing import Callable, Tuple
import argparse
import time

import fsspec
import xarray as xr
import zarr
from distributed import Client
from toolz import curry

from distributed import Client
from dask_kubernetes import KubeCluster
from gcs_aio_mapper import GCSMapperAio


import vcm
from fv3net.pipelines import list_timesteps

DIR = os.path.dirname(__file__)
WORKER_YAML = os.path.join(DIR, "worker-spec.yml")

def setup(*args):
    logging.basicConfig(level=logging.DEBUG)

setup()

CATEGORIES = (
    "fv_srf_wnd_coarse.res",
    "sfc_data_coarse",
    "fv_tracer_coarse.res",
    "fv_core_coarse.res",
)


def _file(url: str, time: str, category: str, tile: int) -> str:
    return os.path.join(url, time, f"{time}.{category}.tile{tile}.nc")


def get_timestep(fs, url, time, category, tile) -> xr.Dataset:
    location = _file(url, time, category, tile)
    logging.info(f"Opening {location}")
    with fs.open(location, "rb") as f:
        ds = xr.open_dataset(f).load()
        return vcm.standardize_metadata(ds)


def insert_timestep(
    out_path: str,
    get: Callable[[str, str, int], xr.Dataset],
    key: Tuple[str, str, int],
):
    
    # The ZarrMapping doesn't serialize so reinit it here.
    output = vcm.ZarrMapping(_get_store(out_path))
    time, category, tile = key
    data = get(time, category, tile)
    logging.info(f"Setting data at {time} for {category} tile {tile}")
    output[time, tile] = data
    output.flush()


def get_schema(fs: fsspec.AbstractFileSystem, url: str) -> xr.Dataset:
    logging.info(f"Grabbing schema from {url}")
    with fs.open(url, "rb") as f:
        return vcm.standardize_metadata(xr.open_dataset(f))

def test_call():
    return _call_me()

def _call_me():
    return "Hello"


def _get_store(output: str):
    if output.startswith("gs://"):
        # use asynchronous GCSMapper class that I wrote with my free time -- Noah
        return GCSMapperAio(output, cache_size=10)
    else:
        return fsspec.get_mapper(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help="root directory of time steps")
    parser.add_argument('output', help="Location of output zarr")
    parser.add_argument("-s", "--n-steps", default=-1, type=int)
    parser.add_argument("-n", "--workers", default=1, type=int)

    args = parser.parse_args()

    logging.info("Building dask cluster and client")
    cluster = KubeCluster.from_yaml(WORKER_YAML)
    cluster.scale_up(args.workers)  # specify number of nodes explicitly
    client = Client(cluster)
    # client = Client()
    client.register_worker_callbacks(setup)
    logging.info("testing client function call")
    res = client.submit(test_call)
    logging.info(res.result())

    # TODO Parameterize these settings
    times = list_timesteps(args.url)
    if args.n_steps != -1:
        times = times[:args.n_steps]

    logging.info("Getting the schema")
    fs = fsspec.filesystem("gs")
    categories = CATEGORIES
    tiles = [1, 2, 3, 4, 5, 6]

    # get schema for first timestep
    time: str = times[0]
    schema = xr.merge(
        [get_schema(fs, _file(args.url, time, category, tile=1)) for category in categories]
    )
    print("Schema:")
    print(schema)

    logging.info("Initializing output zarr")
    store = _get_store(args.output)
    output_m = vcm.ZarrMapping.from_schema(
        store, schema, dims=["time", "tile"], coords={"tile": tiles, "time": times}
    )
    store.flush()

    load_timestep = curry(get_timestep, fs, args.url)
    map_fn = curry(insert_timestep, args.output, load_timestep)

    logging.info("Mapping job")
    results = client.map(map_fn, list(product(times, categories, tiles)), retries=3, resources={'MEMORY': 3.0e9})
    # wait for all results to complete
    try:
        client.gather(results)
    except:
        print("exception!")
        time.sleep(10)
    else:
        logging.info("Job completed!")

import zarr
from gcs_aio_mapper import GCSMapperAio
from gcsfs import GCSMap
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
n = 25

def build_gs_async():
    logging.info("\n\n\nCalling build_gs_async\n\n\n:")
    store = GCSMapperAio("gs://vcm-ml-data/tmp/test.zarr", cache_size=n)
    g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
    for i in range(n):
        g[i] = i
    store.flush()


def build_gs():
    logging.info("\n\n\nCalling build_gs_gcsfs\n\n\n:")
    store = GCSMap("gs://vcm-ml-data/tmp/test.zarr")
    g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
    for i in range(n):
        g[i] = i


build_gs_async()
build_gs()

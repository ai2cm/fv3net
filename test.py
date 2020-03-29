import zarr
from gcsfs_aoi import GCSMapperAio
from gcsfs import GCSMap

n = 25

def build_gs_async():
    store = GCSMapperAio("gs://vcm-ml-data/tmp/test.zarr", cache_size=n)
    for i in range(n):
        store[str(i)] = i


def build_gs():
    store = GCSMap("gs://vcm-ml-data/tmp/test.zarr")
    g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
    for i in range(n):
        store[str(i)] = i
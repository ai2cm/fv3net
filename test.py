import zarr
from gcsfs_aoi import GCSFSMapperAoi
from gcsfs import GCSMap

n = 25

def build_gs_async():
    store = GCSFSMapperAoi("gs://vcm-ml-data/tmp/test.zarr", cache_size=n)
    g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
    for i in range(n):
        g[i] = i


def build_gs():
    store = GCSMap("gs://vcm-ml-data/tmp/test.zarr")
    g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
    for i in range(n):
        g[i] = i
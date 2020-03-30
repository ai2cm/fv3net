Async GCS Mapper
================

Example::

    import zarr
    from gcs_aio_mapper import GCSMapperAio
    from gcsfs import GCSMap
    import logging

    logging.basicConfig(level=logging.DEBUG)
    n = 25

    def build_gs_async():
        store = GCSMapperAio("gs://vcm-ml-data/tmp/test.zarr", cache_size=n)
        g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
        for i in range(n):
            g[i] = i
        store.flush()


    def build_gs():
        store = GCSMap("gs://vcm-ml-data/tmp/test.zarr")
        g = zarr.open_array(store, shape=(n,), chunks=(3,), mode="w")
        for i in range(n):
            g[i] = i

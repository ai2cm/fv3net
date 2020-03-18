import fsspec
import zarr

url = "gs://vcm-ml-data/testing-noah/one-step/big.zarr/"
m = fsspec.get_mapper(url)
g = zarr.open_group(m, mode='r')

print(g['air_temperature'][:].std())


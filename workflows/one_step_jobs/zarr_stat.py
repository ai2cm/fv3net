import fsspec
import zarr

url = "gs://vcm-ml-data/testing-noah/one-step/big.zarr/"
m = fsspec.get_mapper(url)
g = zarr.open_group(m, mode='r')
for variable in g:

    print(variable, g[variable].nbytes/1e9, "GB")

print(g['air_temperature'][:].std())


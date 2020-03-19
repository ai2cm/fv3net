import fsspec
import xarray as xr
import zarr

url = "gs://vcm-ml-data/testing-noah/one-step/big.zarr/"
m = fsspec.get_mapper(url)
ds = xr.open_zarr(m)

print("output structure:")
print()
for root, dirname, filename in fsspec.filesystem("gs").walk(
    "gs://vcm-ml-data/testing-noah/one-step"
):
    if not "big.zarr" in root:
        for name in filename:
            print(f"{root}/{name}")
        for name in dirname:
            print(f"{root}/{name}/")

print()
print("big.zarr info:")
print()
print(ds.info())
print(ds.air_temperature.std().compute())

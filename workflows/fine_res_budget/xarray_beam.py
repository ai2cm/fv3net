import xarray as xr
import fsspec
import apache_beam as beam

restart_url = (
    "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr"
)
restarts = xr.open_zarr(fsspec.get_mapper(restart_url), consolidated=True)

p = beam.Pipeline()
p | beam.Create([restarts])
p.run()
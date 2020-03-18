import xarray as xr
import numpy as np

CHUNK = {'time': 1, 'tile': -1}

begin = xr.open_zarr("output_dir/begin_physics.zarr")
before = xr.open_zarr("output_dir/before_physics.zarr")
after = xr.open_zarr("output_dir/after_physics.zarr")

# make the time dims consistent
time = begin.time
before = before.drop('time')
after = after.drop('time')
begin = begin.drop('time')

# concat data
dt = np.timedelta64(15, 'm')
time = np.arange(len(time)) * dt
ds = xr.concat([begin, before, after], dim='step').assign_coords(step=['begin', 'after_dynamics', 'after_physics'], time=time)
ds.chunk(CHUNK).to_zarr("post_processed.zarr", mode='w')

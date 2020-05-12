import xarray as xr
import logging

import budgets
import dask
dask.config.set(scheduler="single-threaded")

merged = xr.open_zarr("./local.zarr")

c48_budget = budgets.compute_recoarsened_budget(merged, factor=8)

logging.info("Saving to disk with dask")
c48_budget.to_netcdf("c48_budget.nc")

import merge_restarts_and_diags
import budgets
import logging


physics_url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/"
restart_url = "gs://vcm-ml-data/2019-12-02-40-day-X-SHiELD-simulation-C384-restart-files.zarr/"

# TODO remove this function, not good to open and merge data in the same step.
merged = merge_restarts_and_diags.open_merged_data(
    restart_url,
    physics_url
)

logging.info("Computing coarse-grained budgets")
c48_budget = budgets.compute_recoarsened_budget(merged, factor=8)

logging.info("Saving to disk with dask")
c48_budget.to_netcdf("c48_budget.nc")
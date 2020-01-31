import xarray as xr
import intake
import os
import shutil
from vcm import coarsen
from vcm.cloud import gsutil
import logging

logging.basicConfig(level=logging.INFO)

catalog = intake.open_catalog("../../catalog.yml")
diag_path = catalog["40day_c384_diags_time_avg"]
diag_path = diag_path[:-1] if diag_path[-1] == "/" else diag_path

# write coarsened C48 diags to same dir as high res diags
c48_filename = "C48_gfsphysics_15min_coarse.zarr"
output_path = os.path.join(os.path.dirname(diag_path), c48_filename)

diags = catalog["40day_c384_diags_time_avg"].to_dask()

logging.info(f"Size of diagnostic data:  {diags.nbytes / 1e9:.2f} GB")

# rename the dimensions approriately
grid384 = diags[
    [
        "grid_lat_coarse",
        "grid_latt_coarse",
        "grid_lon_coarse",
        "grid_lont_coarse",
        "area_coarse",
    ]
]

diags384 = xr.merge([diags, grid384]).rename(
    {
        "grid_lat_coarse": "latb",
        "grid_latt_coarse": "lat",
        "grid_lon_coarse": "lonb",
        "grid_lont_coarse": "lon",
        "grid_xt_coarse": "grid_xt",
        "grid_yt_coarse": "grid_yt",
        "grid_x_coarse": "grid_x",
        "grid_y_coarse": "grid_y",
    }
)

# coarsen the data
diags48 = coarsen.weighted_block_average(
    diags384,
    diags384["area_coarse"],
    x_dim="grid_xt",
    y_dim="grid_yt",
    coarsening_factor=8,
)

diags48.to_zarr(c48_filename, mode="w")
gsutil.copy(c48_filename, output_path)

logging.info(f"Done writing coarsened C48 zarr to {output_path}")
shutil.rmtree(c48_filename)

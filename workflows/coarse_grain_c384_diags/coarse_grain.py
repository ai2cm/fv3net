import xarray as xr
import intake
import fsspec
from vcm import coarsen
import logging

logging.basicConfig(level=logging.INFO)

catalog = intake.open_catalog("../../catalog.yml")
OUTPUT_PATH = catalog["40day_c48_diags"].urlpath

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

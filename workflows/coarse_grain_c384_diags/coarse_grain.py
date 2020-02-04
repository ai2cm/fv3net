import intake
import logging
import os
import shutil
import xarray as xr

from vcm import coarsen
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
)

logging.basicConfig(level=logging.INFO)


C48_OUTPUT_FILENAME = "C48_gfsphysics_15min_coarse.zarr"
HIRES_DATA_VARS = [
    "LHTFLsfc_coarse",
    "SHTFLsfc_coarse",
    "PRATEsfc_coarse",
    "DSWRFtoa_coarse",
]

catalog = intake.open_catalog("../../catalog.yml")
diag_path = catalog["40day_c384_diags_time_avg"].urlpath
diag_path = diag_path[:-1] if diag_path[-1] == "/" else diag_path

# write coarsened C48 diags to same dir as high res diags
output_path = os.path.join(os.path.dirname(diag_path), C48_OUTPUT_FILENAME)

diags = catalog["40day_c384_diags_time_avg"].to_dask()
logging.info(f"Size of diagnostic data:  {diags.nbytes / 1e9:.2f} GB")

# rename the dimensions appropriately
grid384 = diags[
    [
        "grid_lat_coarse",
        "grid_latt_coarse",
        "grid_lon_coarse",
        "grid_lont_coarse",
        "area_coarse",
    ]
]

diags384 = xr.merge([diags[HIRES_DATA_VARS], grid384]).rename(
    {
        "grid_lat_coarse": VAR_LAT_OUTER,
        "grid_latt_coarse": VAR_LAT_CENTER,
        "grid_lon_coarse": VAR_LON_OUTER,
        "grid_lont_coarse": VAR_LON_CENTER,
        "grid_xt_coarse": COORD_X_CENTER,
        "grid_yt_coarse": COORD_Y_CENTER,
        "grid_x_coarse": COORD_X_OUTER,
        "grid_y_coarse": COORD_Y_OUTER,
    }
)

# coarsen the data
diags48 = coarsen.weighted_block_average(
    diags384[HIRES_DATA_VARS],
    diags384["area_coarse"],
    x_dim=COORD_X_CENTER,
    y_dim=COORD_Y_CENTER,
    coarsening_factor=8,
)

diags48 = diags48.unify_chunks()

diags48.to_zarr(C48_OUTPUT_FILENAME, mode="w", consolidated=True)
gsutil.copy(C48_OUTPUT_FILENAME, output_path)

logging.info(f"Done writing coarsened C48 zarr to {output_path}")
shutil.rmtree(C48_OUTPUT_FILENAME)

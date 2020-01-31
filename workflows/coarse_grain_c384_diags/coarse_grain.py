from datetime import timedelta
import xarray as xr
import intake
import os
import shutil
from vcm import coarsen
from vcm.cloud import gsutil
import logging

HIRES_DATA_VARS = [
    "LHTFLsfc_coarse",
    "SHTFLsfc_coarse",
    "PRATEsfc_coarse",
    "DSWRFtoa_coarse",
]

logging.basicConfig(level=logging.INFO)

catalog = intake.open_catalog("../../catalog.yml")
diag_path = catalog["40day_c384_diags_time_avg"].urlpath
diag_path = diag_path[:-1] if diag_path[-1] == "/" else diag_path

# write coarsened C48 diags to same dir as high res diags
c48_filename = "C48_gfsphysics_15min_coarse.zarr"
output_path = os.path.join(os.path.dirname(diag_path), c48_filename)

diags = catalog["40day_c384_diags_time_avg"].to_dask()

logging.info(f"Size of diagnostic data:  {diags.nbytes / 1e9:.2f} GB")


def _round_time(t):
    """ The high res data timestamps are often +/- a few 1e-2 seconds off the
    initialization times of the restarts, which makes it difficult to merge on
    time. This rounds time to the nearest second, assuming the init time is at most
    1 sec away from a round minute.

    Args:
        t: datetime or cftime object

    Returns:
        datetime or cftime object rounded to nearest minute
    """
    if t.second == 0:
        return t.replace(microsecond=0)
    elif t.second == 59:
        return t.replace(microsecond=0) + timedelta(seconds=1)
    else:
        raise ValueError(
            f"Time value > 1 second from 1 minute timesteps for "
            "C48 initialization time {t}. Are you sure you're joining "
            "the correct high res data?"
        )


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

diags384 = xr.merge([diags[HIRES_DATA_VARS], grid384]).rename(
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

# Drop grid coords
diags48 = (
    diags48[HIRES_DATA_VARS]
    .assign_coords({"time": [_round_time(t) for t in diags48.time.values]})
    .unify_chunks()
)

diags48.to_zarr(c48_filename, mode="w")
gsutil.copy(c48_filename, output_path)

logging.info(f"Done writing coarsened C48 zarr to {output_path}")
shutil.rmtree(c48_filename)

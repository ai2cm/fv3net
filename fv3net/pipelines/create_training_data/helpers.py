from datetime import timedelta
import fsspec
import logging
import os
import xarray as xr

from vcm.fv3_restarts import _split_url
from vcm.cubedsphere.constants import (
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
    INIT_TIME_DIM,
    TIME_FMT
)


GRID_VAR_MAP = {
    "grid_lat_coarse": VAR_LAT_OUTER,
    "grid_lon_coarse": VAR_LON_OUTER,
    "grid_latt_coarse": VAR_LAT_CENTER,
    "grid_lont_coarse": VAR_LON_CENTER,
    "grid_xt_coarse": COORD_X_CENTER,
    "grid_yt_coarse": COORD_Y_CENTER,
    "grid_x_coarse": COORD_X_OUTER,
    "grid_y_coarse": COORD_Y_OUTER,
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _path_from_first_timestep(ds, train_test_labels=None):
    """ Uses first init time as zarr filename, and appends a 'train'/'test' subdir
    if a dict of labels is provided

    Args:
        ds:
        train_test_labels: optional dict with keys ["test", "train"] and values lists of
            timestep strings that go to each set

    Returns:
        path in args.gcs_output_dir to write the zarr to
    """
    timestep = min(ds[INIT_TIME_DIM].values).strftime(TIME_FMT)
    if isinstance(train_test_labels, dict):
        try:
            if timestep in train_test_labels["train"]:
                train_test_subdir = "train"
            elif timestep in train_test_labels["test"]:
                train_test_subdir = "test"
        except KeyError:
            logger.warning(
                "train_test_labels dict does not have keys ['train', 'test']."
                "Will write zarrs directly to gcs_output_dir."
            )
            train_test_subdir = ""
    else:
        logger.info(
            "No train_test_labels dict provided."
            "Will write zarrs directly to gcs_output_dir."
        )
        train_test_subdir = ""
    return os.path.join(train_test_subdir, timestep + ".zarr")


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
        raise ValueError(f"Time value > 1 second from 1 minute timesteps for "
                         "C48 initialization time {t}. Are you sure you're joining "
                         "the correct high res data?")


def load_c384_diag(c384_data_path, init_times):
    protocol, path = _split_url(c384_data_path)
    fs = fsspec.filesystem(protocol)
    ds_c384 = xr.open_zarr(fs.get_mapper(c384_data_path))
    ds_c384 = (
        ds_c384.rename(GRID_VAR_MAP)
        .rename({"time": "initialization_time"})
        .assign_coords(
            {
                "tile": range(6),
                INIT_TIME_DIM: [_round_time(t) for t in ds_c384.time.values],
            }
        )
        .sel({INIT_TIME_DIM: init_times})
    )
    return ds_c384


def add_coarsened_features(ds_c48):
    ds_features = ds_c48.assign(
        {
            "insolation": ds_c48[
                "DSWRFtoa_coarse"
            ],  # downward longwave at TOA << shortwave
            "LHF": ds_c48["LHTFLsfc_coarse"],
            "SHF": ds_c48["SHTFLsfc_coarse"],
            "precip_sfc": ds_c48["PRATEsfc_coarse"],
        }
    )
    return ds_features

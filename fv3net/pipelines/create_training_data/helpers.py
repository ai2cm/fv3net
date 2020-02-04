from datetime import timedelta
import fsspec
import logging
import os
import xarray as xr

from vcm.fv3_restarts import _split_url
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    TIME_FMT,
    TILE_COORDS,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def _path_from_first_timestep(ds, train_test_labels=None):
    """ Uses first init time as zarr filename, and appends a 'train'/'test' subdir
    if a dict of labels is provided

    Args:
        ds: input dataset
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


def _set_relative_forecast_time_coord(ds):
    delta_t_forecast = (
        ds[FORECAST_TIME_DIM].values[-1] - ds[FORECAST_TIME_DIM].values[-2]
    )
    ds.reset_index([FORECAST_TIME_DIM], drop=True)
    return ds.assign_coords(
        {FORECAST_TIME_DIM: [timedelta(seconds=0), delta_t_forecast]}
    )


def load_diag(diag_data_path, init_times):
    protocol, path = _split_url(diag_data_path)
    fs = fsspec.filesystem(protocol)
    ds_diag = xr.open_zarr(fs.get_mapper(diag_data_path), consolidated=True).rename(
        {"time": INIT_TIME_DIM}
    )
    ds_diag = ds_diag.assign_coords(
        {
            INIT_TIME_DIM: [_round_time(t) for t in ds_diag[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )
    return ds_diag.sel({INIT_TIME_DIM: init_times})

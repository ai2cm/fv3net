from datetime import timedelta
import logging
import os
import pandas as pd
import xarray as xr

from vcm.fv3_restarts import open_diagnostic
from vcm.cloud.fsspec import get_fs
from vcm.convenience import round_time
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    TIME_FMT,
    TILE_COORDS,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _convert_forecast_time_to_timedelta(ds, forecast_time_dim):
    timedelta_coords = ds[forecast_time_dim].astype("timedelta64[s]")
    return ds.assign_coords({forecast_time_dim: timedelta_coords})


def _path_from_first_timestep(ds, init_time_dim, time_fmt, train_test_labels=None):
    """ Uses first init time as zarr filename, and appends a 'train'/'test' subdir
    if a dict of labels is provided

    Args:
        ds: input dataset
        train_test_labels: optional dict with keys ["test", "train"] and values lists of
            timestep strings that go to each set

    Returns:
        path in args.gcs_output_dir to write the zarr to
    """
    timestep = min(ds[init_time_dim].values).strftime(time_fmt)
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


def load_hires_prog_diag(diag_data_path, init_times):
    """Loads coarsened diagnostic variables from the prognostic high res run.
    
    Args:
        diag_data_path (str): path to directory containing coarsened high res
            diagnostic data
        init_times (list(datetime)): list of datetimes to filter diagnostic data to
    
    Returns:
        xarray dataset: prognostic high res diagnostic variables
    """
    fs = get_fs(diag_data_path)
    ds_diag = xr.open_zarr(fs.get_mapper(diag_data_path), consolidated=True).rename(
        {"time": INIT_TIME_DIM}
    )
    ds_diag = ds_diag.assign_coords(
        {
            INIT_TIME_DIM: [round_time(t) for t in ds_diag[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )
    return ds_diag.sel({INIT_TIME_DIM: init_times})


def load_train_diag(top_level_dir, init_times):
    """ Loads diag variables from the run dirs that correspond to init_times
    
    Args:
        top_level_dir (str): directory that contains the individual one step rundirs
        init_times (list[datetime]): list of datetimes to load diags for
    """
    init_times = sorted(init_times)
    time_dim_index = pd.Index(init_times, name=INIT_TIME_DIM)
    one_step_diags = []
    for init_time in init_times:
        run_dir = os.path.join(top_level_dir, init_time.strftime(TIME_FMT))
        ds_diag = open_diagnostic(run_dir, "sfc_dt_atmos").isel(time=0)
        one_step_diags.append(ds_diag.squeeze().drop("time"))
    return xr.concat([ds for ds in one_step_diags], time_dim_index)


def _rename_centered_xy_coords(cell_centered_da, edge_dim, center_dim):
    """
    Args:
        cell_centered_da: data array that got shifted from edges to cell centers
    Returns:
        same input array with dims renamed to corresponding cell center dims
    """
    # cell_centered_da[edge_dim] = cell_centered_da[edge_dim] - 1
    cell_centered_da = cell_centered_da.rename({edge_dim: center_dim})
    return cell_centered_da


def _shift_edge_var_to_center(edge_var: xr.DataArray, edge_to_center_dims):
    """
    Args:
        edge_var: variable that is defined on edges of grid, e.g. u, v

    Returns:
        data array with the original variable at cell center
    """
    edge_dims = list(edge_to_center_dims.keys())
    for dim_to_recenter in [
        edge_dim for edge_dim in edge_dims if edge_dim in edge_var.dims
    ]:

        return _rename_centered_xy_coords(
            0.5
            * (edge_var + edge_var.shift({dim_to_recenter: 1})).isel(
                {dim_to_recenter: slice(1, None)}
            ),
            edge_dim=dim_to_recenter,
            center_dim=edge_to_center_dims[dim_to_recenter],
        )
    else:
        raise ValueError(
            "Variable to shift to center must be centered on one horizontal axis and "
            "edge-valued on the other."
        )

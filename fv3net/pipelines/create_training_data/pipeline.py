import logging
import os
from typing import Mapping, Sequence, Tuple

import apache_beam as beam
import dask
import fsspec
import numpy as np
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from vcm import parse_datetime_from_str, safe
from vcm.calc import apparent_source
from vcm.convenience import round_time
from vcm.cubedsphere.constants import INIT_TIME_DIM, TILE_COORDS

from . import helpers

dask.config.set(scheduler="single-threaded")

logger = logging.getLogger(__name__)

TIME_FMT = "%Y%m%d.%H%M%S"
GRID_SPEC_FILENAME = "grid_spec.zarr"

# forecast time step used to calculate the FV3 run tendency
FORECAST_TIME_INDEX_FOR_C48_TENDENCY = 13
# forecast time step used to calculate the high res tendency
FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY = FORECAST_TIME_INDEX_FOR_C48_TENDENCY


def _load_pair(timesteps, ds, init_time_dim):
    yield ds.sel({init_time_dim: timesteps})


def run(
    ds: xr.Dataset,
    ds_diag: xr.Dataset,
    output_dir: str,
    pipeline_args: Sequence[str],
    names: dict,
    timesteps: Mapping[str, Sequence[Tuple[str, str]]],
):
    """ Divide full one step output data into batches to be sent
    through a beam pipeline, which writes training/test data zarrs

    Args:
        ds: The one-step runs xarray dataset
        ds_diag: The coarsened diagnostic data from ShiELD at C48 resolution
        output_dir: the output location in GCS or local
        pipeline_args: argument to be handled by apache beam.
        names: Configuration for output variable names
        timesteps:  a collection of time-step pairs to use for testing and training.
            For example::

                {"train": [("20180601.000000", "20180601.000000"), ...], "test: ...}
    """
    train_test_labels = _train_test_labels(timesteps)
    timestep_pairs = timesteps["train"] + timesteps["test"]

    logger.info(f"Processing {len(timestep_pairs)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        # TODO this code will be cleaner if we get rid of "data batches" as a concept
        # also it would cleaner to have separately loading piplines for each data source
        # that are merged by a beam.CoGroupBY operation.
        # currently, there is no place easy place for me to put a load operation
        # and check for NaNs
        (
            p
            | beam.Create(timestep_pairs)
            | "SelectInitialTimes" >> beam.ParDo(_load_pair, ds, names["init_time_dim"])
            | "TimeDimToDatetime"
            >> beam.Map(_str_time_dim_to_datetime, time_dim=names["init_time_dim"])
            | "AddPhysicsTendencies"
            >> beam.Map(
                _add_physics_tendencies,
                physics_tendency_names=names["physics_tendency_names"],
                forecast_time_dim=names["forecast_time_dim"],
                step_dim=names["step_time_dim"],
                coord_before_physics=names["coord_before_physics"],
                coord_after_physics=names["coord_after_physics"],
            )
            | "SelectStep"
            >> beam.Map(
                lambda ds: ds.sel(
                    {names["step_time_dim"]: names["step_for_state"]}
                ).drop(names["step_time_dim"])
            )
            | "PreprocessOneStepData"
            >> beam.Map(
                _preprocess_one_step_data,
                flux_vars=names["diag_vars"],
                suffix_coarse_train=names["suffix_coarse_train"],
                forecast_time_dim=names["forecast_time_dim"],
                wind_vars=(names["var_x_wind"], names["var_y_wind"]),
                edge_to_center_dims=names["edge_to_center_dims"],
            )
            | "AddApparentSources"
            >> beam.Map(
                _add_apparent_sources,
                init_time_dim=names["init_time_dim"],
                forecast_time_dim=names["forecast_time_dim"],
                var_source_name_map=names["var_source_name_map"],
                tendency_tstep_onestep=FORECAST_TIME_INDEX_FOR_C48_TENDENCY,
                tendency_tstep_highres=FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY,
            )
            | "SelectOneStepCols" >> beam.Map(lambda x: x[list(names["one_step_vars"])])
            | "MergeHiresDiagVars"
            >> beam.Map(
                _merge_hires_data,
                ds_diag,
                renamed_high_res_vars=names["renamed_high_res_data_variables"],
                init_time_dim=names["init_time_dim"],
                renamed_dims=names["renamed_dims"],
            )
            | "LoadData" >> beam.Map(lambda ds: ds.load())
            | "FilterOnNaN" >> beam.Filter(_no_nan_values)
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                init_time_dim=names["init_time_dim"],
                gcs_output_dir=output_dir,
                train_test_labels=train_test_labels,
            )
        )


def _train_test_labels(timesteps):
    train_test_labels = {}
    for key, pairs_list in timesteps.items():
        train_test_labels[key] = []
        for pair in pairs_list:
            train_test_labels[key].append(pair[0])
    return train_test_labels


def _no_nan_values(ds):
    nans = np.isnan(ds)
    nan_in_data_var = {key: True in nans[key].values for key in nans}
    if any(nan_present is True for var, nan_present in nan_in_data_var.items()):
        nan_vars = [var for var in nan_in_data_var if nan_in_data_var[var] is True]
        logger.error(f"NaNs detected in data: {nan_vars}")
        return False
    return ds


def _str_time_dim_to_datetime(ds, time_dim):
    datetime_coords = [
        parse_datetime_from_str(time_str) for time_str in ds[time_dim].values
    ]
    return ds.assign_coords({time_dim: datetime_coords})


def _add_physics_tendencies(
    ds,
    physics_tendency_names,
    forecast_time_dim,
    step_dim,
    coord_before_physics,
    coord_after_physics,
):
    # physics timestep is same as forecast time step [s]
    dt = ds[forecast_time_dim].values[1]
    for var, tendency in physics_tendency_names.items():
        ds[tendency] = (
            ds[var].sel({step_dim: coord_after_physics})
            - ds[var].sel({step_dim: coord_before_physics})
        ) / dt

    return ds


def _preprocess_one_step_data(
    ds,
    flux_vars,
    suffix_coarse_train,
    forecast_time_dim,
    wind_vars,
    edge_to_center_dims,
):
    renamed_one_step_vars = {
        var: f"{var}_{suffix_coarse_train}"
        for var in flux_vars
        if var in list(ds.data_vars)
    }
    ds = ds.rename(renamed_one_step_vars)
    ds = helpers._convert_forecast_time_to_timedelta(ds, forecast_time_dim)
    # center vars located on cell edges
    for wind_var in wind_vars:
        ds[wind_var] = helpers._shift_edge_var_to_center(
            ds[wind_var], edge_to_center_dims
        )
    return ds


def _add_apparent_sources(
    ds,
    tendency_tstep_onestep,
    tendency_tstep_highres,
    init_time_dim,
    forecast_time_dim,
    var_source_name_map,
):
    for var_name, source_name in var_source_name_map.items():
        ds[source_name] = apparent_source(
            ds[var_name],
            coarse_tstep_idx=tendency_tstep_onestep,
            highres_tstep_idx=tendency_tstep_highres,
            t_dim=init_time_dim,
            s_dim=forecast_time_dim,
        )
    ds = ds.isel(
        {init_time_dim: slice(None, ds.sizes[init_time_dim] - 1), forecast_time_dim: 0}
    ).drop(forecast_time_dim)
    return ds


def _merge_hires_data(
    ds_run, ds_diag, renamed_high_res_vars, init_time_dim, renamed_dims
):

    init_times = ds_run[init_time_dim].values
    ds_diag = ds_diag.rename({"time": INIT_TIME_DIM})
    ds_diag = ds_diag.assign_coords(
        {
            INIT_TIME_DIM: [round_time(t) for t in ds_diag[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )

    diags_c48 = ds_diag.sel({INIT_TIME_DIM: init_times})

    diags_c48 = safe.get_variables(diags_c48, renamed_high_res_vars.keys())

    renamed_dims = {
        dim: renamed_dims[dim] for dim in renamed_dims if dim in diags_c48.dims
    }
    features_diags_c48 = diags_c48.rename({**renamed_high_res_vars, **renamed_dims})
    return xr.merge([ds_run, features_diags_c48])


def _write_remote_train_zarr(
    ds,
    gcs_output_dir,
    init_time_dim,
    time_fmt=TIME_FMT,
    zarr_name=None,
    train_test_labels=None,
):
    """Writes temporary zarr on worker and moves it to GCS

    Args:
        ds: xr dataset for single training batch
        gcs_dest_path: write location on GCS
        zarr_filename: name for zarr, use first timestamp as label
        train_test_labels: optional dict with keys ["test", "train"] and values lists of
            timestep strings that go to each set
    Returns:
        None
    """
    if not zarr_name:
        zarr_name = helpers._path_from_first_timestep(
            ds, init_time_dim, time_fmt, train_test_labels
        )
    output_path = os.path.join(gcs_output_dir, zarr_name)
    mapper = fsspec.get_mapper(output_path)
    ds.to_zarr(mapper, mode="w", consolidated=True)
    logger.info(f"Done writing zarr to {output_path}")

from typing import Sequence, TypeVar, List
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import shutil
import xarray as xr

from . import helpers
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cloud.fsspec import get_fs
from vcm import parse_datetime_from_str
from fv3net import COARSENED_DIAGS_ZARR_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TIME_FMT = "%Y%m%d.%H%M%S"
GRID_SPEC_FILENAME = "grid_spec.zarr"
ZARR_NAME = "big.zarr"

# forecast time step used to calculate the FV3 run tendency
FORECAST_TIME_INDEX_FOR_C48_TENDENCY = 13
# forecast time step used to calculate the high res tendency
FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY = FORECAST_TIME_INDEX_FOR_C48_TENDENCY


def run(args, pipeline_args, names, timesteps):
    """ Divide full one step output data into batches to be sent
    through a beam pipeline, which writes training/test data zarrs
    
    Args:
        args ([arg namespace]): for named args in the main function
        pipeline_args ([arg namespace]): additional args for the pipeline
        names ([dict]): Contains information related to the variable
            and dimension names from the one step output and created
            by the pipeline.
        timesteps([dict]): keys train/test, values are nested list of paired
         timesteps
    """
    fs = get_fs(args.gcs_input_data_path)
    ds_full = xr.open_zarr(
        fs.get_mapper(os.path.join(args.gcs_input_data_path, ZARR_NAME))
    )
    train_test_labels = {key: value[0] for key, value in timesteps.items()}
    timestep_pairs = _split_pairs(ds_full, timesteps, names["init_time_dim"])
    
    chunk_sizes = {
        "tile": 1,
        names["init_time_dim"]: 1,
        names["coord_y_center"]: 24,
        names["coord_x_center"]: 24,
        names["coord_z_center"]: 79,
    }

    logger.info(f"Processing {len(timestep_pairs)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(timestep_pairs)
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
            | "PreprocessOneStepData"
            >> beam.Map(
                _preprocess_one_step_data,
                flux_vars=names["diag_vars"],
                suffix_coarse_train=names["suffix_coarse_train"],
                step_time_dim=names["step_time_dim"],
                forecast_time_dim=names["forecast_time_dim"],
                coord_begin_step=names["coord_begin_step"],
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
                diag_c48_path=args.diag_c48_path,
                coarsened_diags_zarr_name=COARSENED_DIAGS_ZARR_NAME,
                flux_vars=names["diag_vars"],
                suffix_hires=names["suffix_hires"],
                init_time_dim=names["init_time_dim"],
                renamed_dims=names["renamed_dims"],
            )
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                chunk_sizes=chunk_sizes,
                init_time_dim=names["init_time_dim"],
                gcs_output_dir=args.gcs_output_data_dir,
                train_test_labels=train_test_labels,
            )
        )


def _split_pairs(ds_full, timesteps, init_time_dim):
    tstep_pairs = []
    for key, pairs in timesteps.items():
        tstep_pairs += pairs
    ds_pairs = [ds_full.sel(init_time_dim=pair) for pair in tstep_pairs]
    return ds_pairs


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
    step_time_dim,
    coord_begin_step,
    wind_vars,
    edge_to_center_dims,
):
    renamed_one_step_vars = {
        var: f"{var}_{suffix_coarse_train}"
        for var in flux_vars
        if var in list(ds.data_vars)
    }
    try:
        ds = ds.sel({step_time_dim: coord_begin_step}).drop(step_time_dim)
        ds = ds.rename(renamed_one_step_vars)
        ds = helpers._convert_forecast_time_to_timedelta(ds, forecast_time_dim)
        # center vars located on cell edges
        for wind_var in wind_vars:
            ds[wind_var] = helpers._shift_edge_var_to_center(
                ds[wind_var], edge_to_center_dims
            )
        return ds
    except (KeyError, ValueError) as e:
        logger.error(f"Failed step PreprocessTrainData: {e}")


def _add_apparent_sources(
    ds,
    tendency_tstep_onestep,
    tendency_tstep_highres,
    init_time_dim,
    forecast_time_dim,
    var_source_name_map,
):
    try:
        for var_name, source_name in var_source_name_map.items():
            ds[source_name] = apparent_source(
                ds[var_name],
                coarse_tstep_idx=tendency_tstep_onestep,
                highres_tstep_idx=tendency_tstep_highres,
                t_dim=init_time_dim,
                s_dim=forecast_time_dim,
            )
        ds = ds.isel(
            {
                init_time_dim: slice(None, ds.sizes[init_time_dim] - 1),
                forecast_time_dim: 0,
            }
        ).drop(forecast_time_dim)
        return ds
    except (ValueError, TypeError) as e:
        logger.error(f"Failed step CreateTrainingCols: {e}")


def _merge_hires_data(
    ds_run,
    diag_c48_path,
    coarsened_diags_zarr_name,
    flux_vars,
    suffix_hires,
    init_time_dim,
    renamed_dims,
):

    renamed_high_res_vars = {
        **{
            f"{var}_coarse": f"{var}_{suffix_hires}"
            for var in flux_vars
            if var in list(ds_run.data_vars)
        },
        "LHTFLsfc_coarse": f"latent_heat_flux_{suffix_hires}",
        "SHTFLsfc_coarse": f"sensible_heat_flux_{suffix_hires}",
    }
    if not diag_c48_path:
        return ds_run
    try:
        init_times = ds_run[init_time_dim].values
        full_zarr_path = os.path.join(diag_c48_path, coarsened_diags_zarr_name)
        diags_c48 = helpers.load_hires_prog_diag(full_zarr_path, init_times)[
            list(renamed_high_res_vars.keys())
        ]
        renamed_dims = {
            dim: renamed_dims[dim] for dim in renamed_dims if dim in diags_c48.dims
        }
        features_diags_c48 = diags_c48.rename({**renamed_high_res_vars, **renamed_dims})
        return xr.merge([ds_run, features_diags_c48])
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge in features from high res diagnostics: {e}")


def _write_remote_train_zarr(
    ds,
    gcs_output_dir,
    init_time_dim,
    time_fmt=TIME_FMT,
    zarr_name=None,
    train_test_labels=None,
    chunk_sizes=None,
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
    try:
        if not zarr_name:
            zarr_name = helpers._path_from_first_timestep(
                ds, init_time_dim, time_fmt, train_test_labels
            )
            ds = ds.chunk(chunk_sizes)
        output_path = os.path.join(gcs_output_dir, zarr_name)
        ds.to_zarr(zarr_name, mode="w", consolidated=True)
        gsutil.copy(zarr_name, output_path)
        logger.info(f"Done writing zarr to {output_path}")
        shutil.rmtree(zarr_name)
    except (ValueError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to write zarr: {e}")

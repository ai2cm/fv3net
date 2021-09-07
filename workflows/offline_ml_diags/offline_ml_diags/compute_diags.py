import argparse
import dataclasses
import fsspec
import logging
import json
import numpy as np
import os
import sys
from tempfile import NamedTemporaryFile
from vcm.derived_mapping import DerivedMapping
import xarray as xr
import yaml
from typing import Mapping, Sequence, Tuple, List

import diagnostics_utils as utils
import loaders
from vcm import safe, interpolate_to_pressure_levels
import vcm
import fv3fit
from ._plot_input_sensitivity import plot_jacobian, plot_rf_feature_importance
from diagnostics_utils import compute_metrics
from ._helpers import (
    load_grid_info,
    is_3d,
    get_variable_indices,
    insert_r2,
    insert_rmse,
)
from ._select import meridional_transect, nearest_time


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags")

# Additional derived outputs (values) that are added to report if their
# corresponding base ML outputs (keys) are present in model
DERIVED_OUTPUTS_FROM_BASE_OUTPUTS = {"dQ1": "Q1", "dQ2": "Q2"}

# variables that are needed in addition to the model features
DIAGS_NC_NAME = "offline_diagnostics.nc"
DIURNAL_NC_NAME = "diurnal_cycle.nc"
TRANSECT_NC_NAME = "transect_lon0.nc"
METRICS_JSON_NAME = "scalar_metrics.json"
METADATA_JSON_NAME = "metadata.json"
DATASET_DIM_NAME = "dataset"
DERIVATION_DIM_NAME = "derivation"

DELP = "pressure_thickness_of_atmospheric_layer"
PREDICT_COORD = "predict"
TARGET_COORD = "target"


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path", type=str, help=("Local or remote path for reading ML model."),
    )
    parser.add_argument(
        "data_yaml",
        type=str,
        default=None,
        help=("Config file with dataset specifications."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where diagnostic output will be written.",
    )

    parser.add_argument(
        "--snapshot-time",
        type=str,
        default=None,
        help=(
            "Timestep to use for snapshot. Provide a string 'YYYYMMDD.HHMMSS'. "
            "If provided, will use the closest timestep in the test set. If not, will "
            "default to use the first timestep available."
        ),
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help=(
            "Optional path to grid data netcdf. If not provided, defaults to loading "
            "the grid  with the appropriate resolution (given in batch_kwargs) from "
            "the catalog. Useful if you do not have permissions to access the GCS "
            "data in vcm.catalog."
        ),
    )
    parser.add_argument(
        "--grid-resolution",
        type=str,
        default="c48",
        help=(
            "Optional grid resolution used to retrieve grid from the vcm catalog "
            '(e.g. "c48"), ignored if --grid is provided'
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=("Optional n_jobs parameter for joblib.parallel when computing metrics."),
    )
    return parser.parse_args()


def _write_nc(ds: xr.Dataset, output_dir: str, output_file: str):
    output_file = os.path.join(output_dir, output_file)

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        vcm.get_fs(output_dir).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


def _average_metrics_dict(ds_metrics: xr.Dataset) -> Mapping:
    logger.info("Calculating metrics mean and stddev over batches...")
    metrics = {
        var: {
            "mean": float(np.mean(ds_metrics[var].values)),
            "std": float(np.std(ds_metrics[var].values)),
        }
        for var in ds_metrics.data_vars
    }
    return metrics


def _compute_diurnal_cycle(ds: xr.Dataset) -> xr.Dataset:
    diurnal_vars = [
        var
        for var in ds
        if {"tile", "x", "y", "sample", "derivation"} == set(ds[var].dims)
    ]
    return utils.create_diurnal_cycle_dataset(
        ds, ds["lon"], ds["land_sea_mask"], diurnal_vars,
    )


def _compute_summary(ds: xr.Dataset, variables) -> xr.Dataset:
    # ...reduce to diagnostic variables
    if "column_integrated_Q2" in ds:
        net_precip = -ds["column_integrated_Q2"].sel(  # type: ignore
            derivation="target"
        )
    else:
        net_precip = None
    summary = utils.reduce_to_diagnostic(
        ds, ds, net_precipitation=net_precip, primary_vars=variables,
    )
    return summary


def _standardize_names(*args: Sequence[xr.Dataset]):
    renamed = []
    for ds in args:
        renamed.append(ds.rename({var: var.lower() for var in ds}))
    return renamed


def _compute_diagnostics(
    batches: Sequence[xr.Dataset],
    grid: xr.Dataset,
    predicted_vars: List[str],
    n_jobs: int,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    batches_summary, batches_diurnal, batches_metrics = [], [], []

    # for each batch...
    for i, ds in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")

        # ...insert additional variables
        diagnostic_vars_3d = [var for var in predicted_vars if is_3d(ds[var])]
        ds = ds.pipe(utils.insert_column_integrated_vars, diagnostic_vars_3d).load()
        ds.update(grid)
        full_predicted_vars = [var for var in ds if DERIVATION_DIM_NAME in ds[var].dims]

        ds_summary = _compute_summary(ds, diagnostic_vars_3d)
        if DATASET_DIM_NAME in ds.dims:
            sample_dims = ("time", DATASET_DIM_NAME)
        else:
            sample_dims = ("time",)  # type: ignore
        stacked = ds.stack(sample=sample_dims)

        ds_diurnal = _compute_diurnal_cycle(stacked)
        ds_summary["time"] = ds["time"]
        ds_diurnal["time"] = ds["time"]
        ds_metrics = compute_metrics(
            prediction=safe.get_variables(
                ds.sel({DERIVATION_DIM_NAME: PREDICT_COORD}), full_predicted_vars
            ),
            target=safe.get_variables(
                ds.sel({DERIVATION_DIM_NAME: TARGET_COORD}), full_predicted_vars
            ),
            grid=grid,
            delp=ds[DELP],
            n_jobs=n_jobs,
        )

        batches_summary.append(ds_summary.load())
        batches_diurnal.append(ds_diurnal.load())
        batches_metrics.append(ds_metrics.load())
        del ds

    # then average over the batches for each output
    ds_summary = xr.concat(batches_summary, dim="batch")
    ds_diurnal = xr.concat(batches_diurnal, dim="batch").mean(dim="batch")
    ds_metrics = xr.concat(batches_metrics, dim="batch")

    ds_diagnostics, ds_scalar_metrics = _consolidate_dimensioned_data(
        ds_summary, ds_metrics
    )

    ds_scalar_metrics = insert_r2(ds_scalar_metrics)
    ds_diagnostics = ds_diagnostics.pipe(insert_r2).pipe(insert_rmse)
    ds_diagnostics, ds_diurnal, ds_scalar_metrics = _standardize_names(
        ds_diagnostics, ds_diurnal, ds_scalar_metrics
    )
    return ds_diagnostics.mean("batch"), ds_diurnal, ds_scalar_metrics


def _consolidate_dimensioned_data(ds_summary, ds_metrics):
    # moves dimensioned quantities into final diags dataset so they're saved as netcdf
    scalar_metrics = [
        var for var in ds_metrics if ds_metrics[var].size == len(ds_metrics.batch)
    ]
    ds_scalar_metrics = safe.get_variables(ds_metrics, scalar_metrics)
    ds_metrics_arrays = ds_metrics.drop(scalar_metrics)
    ds_diagnostics = ds_summary.merge(ds_metrics_arrays)
    return ds_diagnostics, ds_scalar_metrics


def _get_transect(ds_snapshot: xr.Dataset, grid: xr.Dataset, variables: Sequence[str]):
    ds_snapshot_regrid_pressure = xr.Dataset()
    for var in variables:
        transect_var = [
            interpolate_to_pressure_levels(
                field=ds_snapshot[var].sel(derivation=deriv),
                delp=ds_snapshot[DELP],
                dim="z",
            )
            for deriv in ["target", "predict"]
        ]
        ds_snapshot_regrid_pressure[var] = xr.concat(transect_var, dim="derivation")
    ds_snapshot_regrid_pressure = xr.merge([ds_snapshot_regrid_pressure, grid])
    ds_transect = meridional_transect(
        safe.get_variables(
            ds_snapshot_regrid_pressure, list(variables) + ["lat", "lon"]
        )
    )
    return ds_transect


def insert_prediction(ds: xr.Dataset, ds_pred: xr.Dataset) -> xr.Dataset:
    predicted_vars = ds_pred.data_vars
    nonpredicted_vars = [var for var in ds.data_vars if var not in predicted_vars]
    ds_target = (
        safe.get_variables(ds, [var for var in predicted_vars if var in ds.data_vars])
        .expand_dims(DERIVATION_DIM_NAME)
        .assign_coords({DERIVATION_DIM_NAME: [TARGET_COORD]})
    )
    ds_pred = ds_pred.expand_dims(DERIVATION_DIM_NAME).assign_coords(
        {DERIVATION_DIM_NAME: [PREDICT_COORD]}
    )
    return xr.merge([safe.get_variables(ds, nonpredicted_vars), ds_target, ds_pred])


def _get_predict_function(predictor, variables, grid):
    def transform(ds):
        # Prioritize dataset's land_sea_mask if grid values disagree
        ds = xr.merge(
            [ds, grid], compat="override"  # type: ignore
        )
        derived_mapping = DerivedMapping(ds)
        ds_derived = derived_mapping.dataset(variables)
        ds_prediction = predictor.predict_columnwise(
            safe.get_variables(ds_derived, variables), feature_dim="z"
        )
        return insert_prediction(ds_derived, ds_prediction)

    return transform


def _derived_outputs_from_base_predictions(base_outputs):
    derived_outputs = []
    for base_output in base_outputs:
        if base_output in DERIVED_OUTPUTS_FROM_BASE_OUTPUTS:
            derived_outputs.append(DERIVED_OUTPUTS_FROM_BASE_OUTPUTS[base_output])
    return derived_outputs


def main(args):
    logger.info("Starting diagnostics routine.")

    with fsspec.open(args.data_yaml, "r") as f:
        as_dict = yaml.safe_load(f)
    config = loaders.BatchesLoader.from_dict(as_dict)

    logger.info("Reading grid...")
    if not args.grid:
        # By default, read the appropriate resolution grid from vcm.catalog
        grid = load_grid_info(args.grid_resolution)
    else:
        with fsspec.open(args.grid, "rb") as f:
            grid = xr.open_dataset(f).load()

    logger.info("Opening ML model")
    model = fv3fit.load(args.model_path)

    additional_derived_outputs = _derived_outputs_from_base_predictions(
        model.output_variables
    )
    if len(additional_derived_outputs) > 0:
        model = fv3fit.DerivedModel(
            model, derived_output_variables=additional_derived_outputs
        )

    model_variables = list(set(model.input_variables + model.output_variables + [DELP]))

    output_data_yaml = os.path.join(args.output_path, "data_config.yaml")
    with fsspec.open(args.data_yaml, "r") as f_in, fsspec.open(
        output_data_yaml, "w"
    ) as f_out:
        f_out.write(f_in.read())
    batches = config.load_batches(model_variables)
    predict_function = _get_predict_function(model, model_variables, grid)
    batches = loaders.Map(predict_function, batches)

    # compute diags
    ds_diagnostics, ds_diurnal, ds_scalar_metrics = _compute_diagnostics(
        batches, grid, predicted_vars=model.output_variables, n_jobs=args.n_jobs
    )

    # save model senstivity figures: jacobian (TODO: RF feature sensitivity)
    try:
        plot_jacobian(
            model,
            os.path.join(args.output_path, "model_sensitivity_figures"),  # type: ignore
        )
    except AttributeError:
        try:
            input_feature_indices = get_variable_indices(
                data=batches[0], variables=model.input_variables
            )
            plot_rf_feature_importance(
                input_feature_indices,
                model,
                os.path.join(args.output_path, "model_sensitivity_figures"),
            )
        except AttributeError:
            pass

    if isinstance(config, loaders.BatchesFromMapperConfig):
        mapper = config.load_mapper()
        # compute transected and zonal diags
        snapshot_time = args.snapshot_time or sorted(list(mapper.keys()))[0]
        snapshot_key = nearest_time(snapshot_time, list(mapper.keys()))
        ds_snapshot = predict_function(mapper[snapshot_key])
        transect_vertical_vars = [
            var for var in model.output_variables if is_3d(ds_snapshot[var])
        ]
        ds_transect = _get_transect(ds_snapshot, grid, transect_vertical_vars)

        # write diags and diurnal datasets
        _write_nc(ds_transect, args.output_path, TRANSECT_NC_NAME)

    _write_nc(
        ds_diagnostics, args.output_path, DIAGS_NC_NAME,
    )
    _write_nc(ds_diurnal, args.output_path, DIURNAL_NC_NAME)

    # convert and output metrics json
    metrics = _average_metrics_dict(ds_scalar_metrics)
    with fsspec.open(os.path.join(args.output_path, METRICS_JSON_NAME), "w") as f:
        json.dump(metrics, f, indent=4)

    metadata = {}
    metadata["model_path"] = args.model_path
    metadata["data_config"] = dataclasses.asdict(config)
    with fsspec.open(os.path.join(args.output_path, METADATA_JSON_NAME), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Finished processing dataset diagnostics and metrics.")


if __name__ == "__main__":
    args = _create_arg_parser()
    main(args)

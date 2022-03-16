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

import loaders
from vcm import safe, interpolate_to_pressure_levels
import vcm
import fv3fit
from .compute_diagnostics import compute_diagnostics
from .derived_diagnostics import derived_registry
from ._input_sensitivity import plot_input_sensitivity
from ._helpers import (
    DATASET_DIM_NAME,
    load_grid_info,
    is_3d,
    insert_r2,
    insert_aggregate_bias,
    insert_rmse,
    insert_column_integrated_vars,
)
from ._select import meridional_transect, nearest_time


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags")


INPUT_SENSITIVITY = "input_sensitivity.png"
DIAGS_NC_NAME = "offline_diagnostics.nc"
TRANSECT_NC_NAME = "transect_lon0.nc"
METRICS_JSON_NAME = "scalar_metrics.json"
METADATA_JSON_NAME = "metadata.json"

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


def _standardize_names(*args: xr.Dataset):
    renamed = []
    for ds in args:
        renamed.append(ds.rename({var: str(var).lower() for var in ds}))
    return renamed


def _compute_diagnostics(
    batches: Sequence[xr.Dataset],
    grid: xr.Dataset,
    predicted_vars: List[str],
    n_jobs: int,
) -> Tuple[xr.Dataset, xr.Dataset]:
    batches_summary = []

    # for each batch...
    for i, ds in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")

        # ...insert additional variables
        diagnostic_vars_3d = [var for var in predicted_vars if is_3d(ds[var])]
        ds = ds.pipe(insert_column_integrated_vars, diagnostic_vars_3d).load()

        full_predicted_vars = [var for var in ds if DERIVATION_DIM_NAME in ds[var].dims]
        if "dQ2" in full_predicted_vars or "Q2" in full_predicted_vars:
            full_predicted_vars.append("water_vapor_path")
        prediction = safe.get_variables(
            ds.sel({DERIVATION_DIM_NAME: PREDICT_COORD}), full_predicted_vars
        )
        target = safe.get_variables(
            ds.sel({DERIVATION_DIM_NAME: TARGET_COORD}), full_predicted_vars
        )
        ds_summary = compute_diagnostics(
            prediction, target, grid, ds[DELP], n_jobs=n_jobs
        )
        ds_summary["time"] = ds["time"]

        batches_summary.append(ds_summary.load())
        del ds

    # then average over the batches for each output
    ds_summary = xr.concat(batches_summary, dim="batch")
    ds_summary = insert_r2(ds_summary)
    if DATASET_DIM_NAME in ds_summary.dims:
        ds_summary = insert_aggregate_bias(ds_summary)
    ds_diagnostics, ds_scalar_metrics = _consolidate_dimensioned_data(ds_summary)
    ds_diagnostics = ds_diagnostics.pipe(insert_rmse)
    ds_diagnostics, ds_scalar_metrics = _standardize_names(
        ds_diagnostics, ds_scalar_metrics
    )
    # this is kept as a coord to use in plotting a histogram of test timesteps
    return ds_diagnostics.mean("batch"), ds_scalar_metrics


def _consolidate_dimensioned_data(ds):
    # moves dimensioned quantities into final diags dataset so they're saved as netcdf
    scalar_metrics = [var for var in ds if ds[var].size == len(ds.batch)]
    ds_scalar_metrics = safe.get_variables(ds, scalar_metrics)
    ds_metrics_arrays = ds.drop(scalar_metrics)
    ds_diagnostics = ds.merge(ds_metrics_arrays)
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
        ds_prediction = predictor.predict(safe.get_variables(ds_derived, variables))
        return insert_prediction(ds_derived, ds_prediction)

    return transform


def _get_data_mapper_if_exists(config):
    if isinstance(config, loaders.BatchesFromMapperConfig):
        return config.load_mapper()
    else:
        return None


def _variables_to_load(model):
    return list(
        set(list(model.input_variables) + list(model.output_variables) + [DELP])
    )


def _add_derived_diagnostics(ds):
    merged = xr.merge([ds, derived_registry.compute(ds, n_jobs=1)])
    return merged.assign_attrs(ds.attrs)


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
            grid = xr.open_dataset(f, engine="h5netcdf").load()

    logger.info("Opening ML model")
    model = fv3fit.load(args.model_path)

    # add Q2 and total water path for PW-Q2 scatterplots and net precip domain averages
    if any(["Q2" in v for v in model.output_variables]):
        model = fv3fit.DerivedModel(model, derived_output_variables=["Q2"])
        model_variables = _variables_to_load(model) + ["water_vapor_path"]
    else:
        model_variables = _variables_to_load(model)

    output_data_yaml = os.path.join(args.output_path, "data_config.yaml")
    with fsspec.open(args.data_yaml, "r") as f_in, fsspec.open(
        output_data_yaml, "w"
    ) as f_out:
        f_out.write(f_in.read())
    batches = config.load_batches(model_variables)

    predict_function = _get_predict_function(model, model_variables, grid)
    batches = loaders.Map(predict_function, batches)

    # compute diags
    ds_diagnostics, ds_scalar_metrics = _compute_diagnostics(
        batches, grid, predicted_vars=model.output_variables, n_jobs=args.n_jobs
    )
    ds_diagnostics = ds_diagnostics.update(grid)

    # save model senstivity figures- these exclude derived variables
    fig_input_sensitivity = plot_input_sensitivity(model, batches[0])
    if fig_input_sensitivity is not None:
        with fsspec.open(
            os.path.join(
                args.output_path, "model_sensitivity_figures", INPUT_SENSITIVITY
            ),
            "wb",
        ) as f:
            fig_input_sensitivity.savefig(f)

    mapper = _get_data_mapper_if_exists(config)
    if mapper is not None:
        snapshot_time = (
            args.snapshot_time
            or sorted(config.kwargs.get("timesteps", list(mapper.keys())))[0]
        )
        snapshot_key = nearest_time(snapshot_time, list(mapper.keys()))
        ds_snapshot = predict_function(mapper[snapshot_key])

        vertical_vars = [
            var for var in model.output_variables if is_3d(ds_snapshot[var])
        ]
        ds_snapshot = insert_column_integrated_vars(ds_snapshot, vertical_vars)
        predicted_vars = [
            var for var in ds_snapshot if "derivation" in ds_snapshot[var].dims
        ]

        # add snapshotted prediction to saved diags.nc
        ds_diagnostics = ds_diagnostics.merge(
            safe.get_variables(ds_snapshot, predicted_vars).rename(
                {v: f"{v}_snapshot" for v in predicted_vars}
            )
        )

        ds_transect = _get_transect(ds_snapshot, grid, vertical_vars)
        _write_nc(ds_transect, args.output_path, TRANSECT_NC_NAME)

    ds_diagnostics = _add_derived_diagnostics(ds_diagnostics)

    _write_nc(
        ds_diagnostics, args.output_path, DIAGS_NC_NAME,
    )

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

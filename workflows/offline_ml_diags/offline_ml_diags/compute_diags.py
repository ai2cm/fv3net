import argparse
import intake
import logging
import json
import numpy as np
import os
import sys
from tempfile import NamedTemporaryFile
import xarray as xr
import yaml
from typing import Mapping, Sequence, Tuple

import diagnostics_utils as utils
import loaders
from vcm import safe
from vcm.cloud import get_fs
from ._metrics import calc_metrics
from . import _model_loaders as model_loaders
from ._mapper import PredictionMapper
from ._helpers import add_net_precip_domain_info


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags")

# variables that are needed in addition to the model features
ADDITIONAL_VARS = ["pressure_thickness_of_atmospheric_layer", "pQ1", "pQ2"]
DIAGS_NC_NAME = "offline_diagnostics.nc"
DIURNAL_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_pQ1",
    "column_integrated_pQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]
SHIELD_DERIVATION_COORD = "coarsened_SHiELD"
KERAS_MODELS = ["DenseModel"]
DIURNAL_NC_NAME = "diurnal_cycle.nc"
METRICS_JSON_NAME = "scalar_metrics.json"


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="*", type=str, help="Location of test data")
    parser.add_argument(
        "config_yml",
        type=str,
        help=("Config file with dataset and variable specifications"),
    )
    parser.add_argument(
        "model_path", type=str, help=("Local or remote path for reading ML model."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="Json file that defines train timestep set.",
    )
    return parser.parse_args()


def _write_nc(ds: xr.Dataset, output_dir: str, output_file: str):
    output_file = os.path.join(output_dir, output_file)

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        get_fs(output_dir).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


def _average_metrics_dict(ds_metrics: xr.Dataset) -> Mapping:
    logger.info("Calculating metrics mean and stddev over batches...")
    metrics = {
        var: {
            "mean": np.mean(ds_metrics[var].values),
            "std": np.std(ds_metrics[var].values),
        }
        for var in ds_metrics.data_vars
    }
    return metrics


def _compute_diags_over_batches(
    ds_batches: Sequence[xr.Dataset], grid: xr.Dataset,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Return a set of diagnostic datasets from a sequence of batched data"""

    batches_summary, batches_diurnal, batches_metrics = [], [], []
    # for each batch...
    for i, ds in enumerate(ds_batches):
        logger.info(f"Working on batch {i} diagnostics ...")
        # ...insert additional variables
        ds = (
            ds.pipe(utils.insert_total_apparent_sources)
            .pipe(utils.insert_column_integrated_vars)
            .pipe(utils.insert_net_terms_as_Qs)
            .load()
        )
        # ...reduce to diagnostic variables
        if SHIELD_DERIVATION_COORD in ds["derivation"].values:
            net_precip_domain_coord = SHIELD_DERIVATION_COORD
        else:
            net_precip_domain_coord = "target"

        ds_summary = utils.reduce_to_diagnostic(
            ds,
            grid,
            net_precipitation=-ds["column_integrated_Q2"].sel(
                derivation=net_precip_domain_coord
            ),
        )
        add_net_precip_domain_info(ds_summary, net_precip_domain_coord)

        # ...compute diurnal cycles
        ds_diurnal = utils.create_diurnal_cycle_dataset(
            ds, grid["lon"], grid["land_sea_mask"], DIURNAL_VARS,
        )
        # ...compute metrics
        ds_metrics = calc_metrics(xr.merge([ds, grid["area"]]))

        batches_summary.append(ds_summary.load())
        batches_diurnal.append(ds_diurnal.load())
        batches_metrics.append(ds_metrics.load())
        del ds
        logger.info(f"Processed batch {i} diagnostics netcdf output.")

    # then average over the batches for each output
    ds_summary = xr.concat(batches_summary, dim="batch")
    ds_diurnal = xr.concat(batches_diurnal, dim="batch").mean(dim="batch")
    ds_metrics = xr.concat(batches_metrics, dim="batch")

    ds_diagnostics, ds_scalar_metrics = _consolidate_dimensioned_data(
        ds_summary, ds_metrics
    )

    return ds_diagnostics.mean("batch"), ds_diurnal, ds_scalar_metrics


def _consolidate_dimensioned_data(ds_summary, ds_metrics):
    # moves dimensioned quantities into final diags dataset so they're saved as netcdf
    metrics_arrays_vars = [var for var in ds_metrics.data_vars if "scalar" not in var]
    ds_metrics_arrays = safe.get_variables(ds_metrics, metrics_arrays_vars)
    ds_diagnostics = ds_summary.merge(ds_metrics_arrays).rename(
        {var: var.replace("/", "-") for var in metrics_arrays_vars}
    )
    return ds_diagnostics, ds_metrics.drop(metrics_arrays_vars)


def _get_base_mapper(args, config: Mapping):
    logger.info("Creating base mapper")
    base_mapping_function = getattr(
        loaders.mappers, config["batch_kwargs"]["mapping_function"]
    )
    data_path = args.data_path
    if len(data_path) == 1:
        data_path = data_path[0]
    return base_mapping_function(
        data_path, **config["batch_kwargs"].get("mapping_kwargs", {})
    )


def _get_model_loader(config: Mapping):
    model_type_str = (
        config.get("model_type", "sklearn_random_forest")
        .replace(" ", "")
        .replace("_", "")
    )
    if ("rf" in model_type_str) or ("randomforest" in model_type_str):
        model_routine = "sklearn"
    elif model_type_str in KERAS_MODELS:
        model_routine = "keras"
    else:
        raise (AttributeError(f"Invalid model_type: {model_type_str}"))
    model_loader_str = (
        "load_sklearn_model" if model_routine == "sklearn" else "load_keras_model"
    )
    model_loader = getattr(model_loaders, model_loader_str)
    loader_kwargs = (
        {"keras_model_type": model_type_str} if model_routine == "keras" else {}
    )
    return model_loader, loader_kwargs


def _get_prediction_mapper(args, config: Mapping):
    base_mapper = _get_base_mapper(args, config)
    logger.info("Opening ML model")
    model_loader, loader_kwargs = _get_model_loader(config)
    model = model_loader(args.model_path, **loader_kwargs)
    model_mapper_kwargs = config.get("model_mapper_kwargs", {})
    if "cos_zenith_angle" in config["input_variables"]:
        model_mapper_kwargs["cos_z_var"] = "cos_zenith_angle"
    logger.info("Creating prediction mapper")
    return PredictionMapper(base_mapper, model, grid=grid, **model_mapper_kwargs)


if __name__ == "__main__":
    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()

    with open(args.config_yml, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Reading grid...")
    cat = intake.open_catalog("catalog.yml")
    grid = cat["grid/c48"].read()
    land_sea_mask = cat["landseamask/c48"].read()
    grid = grid.assign({utils.VARNAMES["surface_type"]: land_sea_mask["land_sea_mask"]})
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])

    if args.timesteps_file:
        logger.info("Reading timesteps file")
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        config["batch_kwargs"]["timesteps"] = timesteps

    # write out config used to generate diagnostics, including model path
    config["model_path"] = args.model_path
    fs = get_fs(args.output_path)
    with fs.open(os.path.join(args.output_path, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    pred_mapper = _get_prediction_mapper(args, config)

    variables = list(
        set(config["input_variables"] + config["output_variables"] + ADDITIONAL_VARS)
    )
    del config["batch_kwargs"]["mapping_function"]
    del config["batch_kwargs"]["mapping_kwargs"]

    ds_batches = loaders.batches.diagnostic_batches_from_mapper(
        pred_mapper, variables, **config["batch_kwargs"],
    )

    # compute diags
    ds_diagnostics, ds_diurnal, ds_scalar_metrics = _compute_diags_over_batches(
        ds_batches, grid
    )

    # write diags and diurnal datasets
    _write_nc(xr.merge([grid, ds_diagnostics]), args.output_path, DIAGS_NC_NAME)
    _write_nc(ds_diurnal, args.output_path, DIURNAL_NC_NAME)

    # convert and output metrics json
    metrics = _average_metrics_dict(ds_scalar_metrics)
    with fs.open(os.path.join(args.output_path, METRICS_JSON_NAME), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Finished processing dataset diagnostics and metrics.")

import argparse
import atexit
import fsspec
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Mapping, Sequence
import xarray as xr
from vcm.cloud import gsutil
import diagnostics_utils.plot as diagplot

DATA_SOURCE_DIM = "data_source"
DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"
DIURNAL_CYCLE_DIM = "local_time_hr"

UNITS_Q1 = "K/s"
UNITS_Q2 = "kg/kg/s"
UNITS_Q1_COL_INT = "W/m^2"
UNITS_Q2_COL_INT = "mm/day"

PROFILE_VARS = ['dQ1', 'pQ1', 'Q1', 'dQ2', 'pQ2', 'Q2']
COLUMN_INTEGRATED_VARS = [
    'column_integrated_dQ1',
    'column_integrated_Q1',
    'column_integrated_dQ2',
    'column_integrated_Q2']

NC_FILE_DIAGS = "offline_diagnostics.nc"
NC_FILE_DIURNAL = "diurnal_cycle.nc"
JSON_FILE_METRICS = "scalar_metrics.json"

GLOBAL_MEAN_VARS = [
    'column_integrated_dQ1_global_mean', 
    'column_integrated_pQ1_global_mean', 
    'column_integrated_Q1_global_mean',
    'column_integrated_dQ2_global_mean', 
    'column_integrated_pQ2_global_mean', 
    'column_integrated_Q2_global_mean']

PROFILE_VARS = ['dQ1', 'pQ1', 'Q1', 'dQ2', 'pQ2', 'Q2']


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags_report")


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help=("Location of diagnostics and metrics data."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--baseline_physics_path",
        type=str,
        default=None,
        help=(
            "Location of baseline physics data. "
            "If omitted, will not add this to the comparison."),
    )
    return parser.parse_args()


def _cleanup_temp_dir(temp_dir):
    logger.info(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


def _open_diagnostics_outputs(
        data_dir,
        diagnostics_nc_name: str = NC_FILE_DIAGS,
        diurnal_nc_name: str = NC_FILE_DIURNAL,
        metrics_json_name: str = JSON_FILE_METRICS,
        ):
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    return ds_diags, ds_diurnal, metrics


def _merge_baseline_comparisons(ds_run, ds_baseline, derivation_dim=DERIVATION_DIM):
    # the offline diags workflow will assign "predict" coord to baseline physics comparison with SHiELD,
    # its "target" (redundant for baseline comparison) and "shield" coordinates
    # are removed and the diurnal cycle/diagnostics are relabeled as "baseline"
    ds_baseline = ds_baseline \
        .sel({derivation_dim: "predict"}) \
        .assign_coords({derivation_dim: "baseline"})
    return xr.concat([ds_baseline, ds_run], dim=derivation_dim)


def _copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def _r2_from_metrics(metrics):



if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()
    
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)



    ds_diags, ds_diurnal, metrics = _open_diagnostics_outputs(args.input_path)

    # TODO: leave this argument out of usage for now.
    # need to make a mapper to open baseline physics
    # to input to diags
    if args.baseline_physics_path:
        ds_diags_baseline, ds_diurnal_baseline, metrics_baseline = \
            _open_diagnostics_outputs(args.baseline_physics_path)
        ds_diags = _merge_baseline_comparisons(ds_diags, ds_diags_baseline)
        ds_diurnal = _merge_baseline_comparisons(ds_diurnal, ds_diurnal_baseline)

    diagplot.plot_profile_vars(
        ds_diags,
        output_dir=temp_output_dir.name,
        profile_vars=PROFILE_VARS,
        derivation_dim=DERIVATION_DIM,
        domain_dim=DOMAIN_DIM,
    )

    diagplot.plot_column_integrated_vars(
        ds_diags,
        output_dir=temp_output_dir.name,
        column_integrated_vars=COLUMN_INTEGRATED_VARS,
        derivation_plot_coords=["target", "predict", "coarsened_SHiELD"]
    )
    for tag, var_group in {
        "Q1_components": ['column_integrated_dQ1', 'column_integrated_Q1'],
        "Q2_components": ['column_integrated_dQ2', 'column_integrated_Q2']
    }.items():
        diagplot.plot_diurnal_cycles(
            ds_diurnal,
            output_dir=temp_output_dir.name,
            tag=tag,
            vars=var_group,
            derivation_plot_coords=["target", "predict", "coarsened_SHiELD"],
    )

    _copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to f{args.output_path}")
    
    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_diurnal.close()
    
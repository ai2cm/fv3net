import argparse
import atexit
import fsspec
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Mapping, Union, Sequence
import xarray as xr

from vcm.cloud import gsutil
import diagnostics_utils.plot as diagplot
from ._metrics import _get_r2_string, _get_bias_string
from report import create_html, insert_report_figure


DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"

PROFILE_VARS = ["dQ1", "Q1", "dQ2", "Q2"]
COLUMN_INTEGRATED_VARS = [
    "column_integrated_dQ1",
    "column_integrated_Q1",
    "column_integrated_dQ2",
    "column_integrated_Q2",
]

NC_FILE_DIAGS = "offline_diagnostics.nc"
NC_FILE_DIURNAL = "diurnal_cycle.nc"
JSON_FILE_METRICS = "scalar_metrics.json"

GLOBAL_MEAN_VARS = [
    "column_integrated_dQ1_global_mean",
    "column_integrated_pQ1_global_mean",
    "column_integrated_Q1_global_mean",
    "column_integrated_dQ2_global_mean",
    "column_integrated_pQ2_global_mean",
    "column_integrated_Q2_global_mean",
]

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
        "input_path", type=str, help=("Location of diagnostics and metrics data."),
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
            "If omitted, will not add this to the comparison."
        ),
    )
    return parser.parse_args()


def _write_report(
        output_dir: str,
        title: str,
        sections: Mapping[str, Sequence[str]],
        metadata: Mapping[str, Union[str, float, int, bool]] = None,
        metrics: Mapping[str, Mapping[str, Union[str, float]]] = None,
        html_header: str = None,
):
    filename = title.replace(" ", "_") + ".html"
    html_report = create_html(
        sections, title, metadata=metadata, metrics=metrics, html_header=html_header)
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


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


def _copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()

    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    ds_diags, ds_diurnal, metrics = _open_diagnostics_outputs(args.input_path)

    report_sections = {}

    # time averaged quantity vertical profiles over land/sea, pos/neg net precip
    for var in PROFILE_VARS:
        fig = diagplot.plot_profile_var(
            ds_diags, var, derivation_dim=DERIVATION_DIM, domain_dim=DOMAIN_DIM,
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Vertical profiles of predicted variables",
            output_dir=temp_output_dir.name)

    # time averaged column integrated quantity maps
    for var in COLUMN_INTEGRATED_VARS:
        fig = diagplot.plot_column_integrated_var(
            ds_diags,
            var,
            derivation_plot_coords=["target", "predict", "coarsened_SHiELD"],
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Time averaged maps",
            output_dir=temp_output_dir.name)

    # column integrated quantity diurnal cycles
    for tag, var_group in {
        "Q1_components": ["column_integrated_dQ1", "column_integrated_Q1"],
        "Q2_components": ["column_integrated_dQ2", "column_integrated_Q2"],
    }.items():
        fig = diagplot.plot_diurnal_cycles(
            ds_diurnal,
            vars=var_group,
            derivation_plot_coords=["target", "predict", "coarsened_SHiELD"],
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{tag}.png",
            section_name="Diurnal cycles of column integrated quantities",
            output_dir=temp_output_dir.name)

    # vertical profiles of bias and RMSE
    pressure_lvl_metrics = [
        var for var in ds_diags.data_vars if "pressure" in ds_diags[var].dims
    ]
    for var in pressure_lvl_metrics:
        fig = diagplot._plot_generic_data_array(
            ds_diags[var], xlabel="pressure [Pa]",
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Pressure level metrics",
            output_dir=temp_output_dir.name)

    # scalar metrics for RMSE and bias
    metrics_formatted = {}
    for var in COLUMN_INTEGRATED_VARS:
        metrics_formatted[var.replace("_", " ")] = {
            "r2": _get_r2_string(metrics, var),
            "bias": _get_bias_string(metrics, var)
        }

    _write_report(
        output_dir=temp_output_dir.name,
        title="ML offline diagnostics",
        sections=report_sections,
        metrics=metrics_formatted,
    )

    _copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # TODO: following PR will add the report saving.
    # Separated that out as I want to make some additions to report,
    # which are getting out of scope.

    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_diurnal.close()

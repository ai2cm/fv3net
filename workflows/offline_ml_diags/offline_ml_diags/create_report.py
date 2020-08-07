import argparse
import atexit
import logging
import sys
import tempfile

from report import insert_report_figure
import diagnostics_utils.plot as diagplot
from ._helpers import (
    get_metric_string,
    write_report,
    open_diagnostics_outputs,
    copy_outputs,
    tidy_title,
    units_from_Q_name,
)


DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"

PRESSURE_LEVEL_METRICS_VARS = [
    "pressure_level-bias-dQ1-predict_vs_target",
    "pressure_level-bias-Q1-predict_vs_target",
    "pressure_level-bias-dQ2-predict_vs_target",
    "pressure_level-bias-Q2-predict_vs_target",
    "pressure_level-r2-dQ1-predict_vs_target",
    "pressure_level-r2-dQ2-predict_vs_target",
    "pressure_level-r2-Q1-predict_vs_target",
    "pressure_level-r2-Q2-predict_vs_target",
]
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

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags_report")


def _cleanup_temp_dir(temp_dir):
    logger.info(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


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


if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()

    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    ds_diags, ds_diurnal, metrics = open_diagnostics_outputs(
        args.input_path,
        diagnostics_nc_name=NC_FILE_DIAGS,
        diurnal_nc_name=NC_FILE_DIURNAL,
        metrics_json_name=JSON_FILE_METRICS,
    )

    report_sections = {}

    # vertical profiles of bias and R2
    for var in PRESSURE_LEVEL_METRICS_VARS:
        ylim = (0, 1) if "r2" in var.lower() else None
        fig = diagplot._plot_generic_data_array(
            ds_diags[var], xlabel="pressure [Pa]", ylim=ylim, title=tidy_title(var)
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Pressure level metrics",
            output_dir=temp_output_dir.name,
        )

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
            output_dir=temp_output_dir.name,
        )

    # time averaged column integrated quantity maps
    for var in COLUMN_INTEGRATED_VARS:
        fig = diagplot.plot_column_integrated_var(
            ds_diags, var, derivation_plot_coords=ds_diags.derivation.values,
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Time averaged maps",
            output_dir=temp_output_dir.name,
        )

    # column integrated quantity diurnal cycles
    for tag, var_group in {
        "Q1_components": ["column_integrated_dQ1", "column_integrated_Q1"],
        "Q2_components": ["column_integrated_dQ2", "column_integrated_Q2"],
    }.items():
        fig = diagplot.plot_diurnal_cycles(
            ds_diurnal, vars=var_group, derivation_plot_coords=["target", "predict"],
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{tag}.png",
            section_name="Diurnal cycles of column integrated quantities",
            output_dir=temp_output_dir.name,
        )

    # scalar metrics for RMSE and bias
    metrics_formatted = {}
    for var in COLUMN_INTEGRATED_VARS:
        metrics_formatted[var.replace("_", " ")] = {
            "r2": get_metric_string(metrics, "r2", var),
            "bias": " ".join(
                [get_metric_string(metrics, "bias", var), units_from_Q_name(var)]
            ),
        }

    write_report(
        output_dir=temp_output_dir.name,
        title="ML offline diagnostics",
        sections=report_sections,
        report_metrics=metrics_formatted,
    )

    copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_diurnal.close()

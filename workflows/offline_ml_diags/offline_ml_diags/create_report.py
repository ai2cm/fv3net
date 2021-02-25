import argparse
from collections import OrderedDict
import os
import atexit
import logging
import sys
import tempfile
from typing import MutableMapping, Sequence, List

import fv3viz
import numpy as np
from report import insert_report_figure
import vcm
from vcm.cloud import get_fs
import diagnostics_utils.plot as diagplot
from ._helpers import (
    get_metric_string,
    write_report,
    open_diagnostics_outputs,
    copy_outputs,
    tidy_title,
    units_from_Q_name,
    column_integrated_metric_names,
    insert_dataset_r2,
    insert_scalar_metrics_r2,
    mse_to_rmse,
)
from ._select import plot_transect


DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"

NC_FILE_DIAGS = "offline_diagnostics.nc"
NC_FILE_DIURNAL = "diurnal_cycle.nc"
NC_FILE_TRANSECT = "transect_lon0.nc"
JSON_FILE_METRICS = "scalar_metrics.json"

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags_report")


def copy_pngs_to_report(input: str, output: str) -> List[str]:
    fs = get_fs(input)
    pngs = fs.glob(os.path.join(input, "*.png"))
    output_pngs = []
    if len(pngs) > 0:
        for png in pngs:
            relative_path = os.path.basename(png)
            fs.get(png, os.path.join(output, relative_path))
            output_pngs.append(relative_path)
    return output_pngs


def _cleanup_temp_dir(temp_dir):
    logger.info(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


def _create_arg_parser() -> argparse.Namespace:
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
        "--commit-sha",
        type=str,
        default=None,
        help=(
            "Commit SHA of fv3net used to create report. Useful for referencing"
            "the version used to train the model."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":

    logger.info("Starting create report routine.")
    args = _create_arg_parser()

    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    ds_diags, ds_diurnal, ds_transect, metrics, config = open_diagnostics_outputs(
        args.input_path,
        diagnostics_nc_name=NC_FILE_DIAGS,
        diurnal_nc_name=NC_FILE_DIURNAL,
        transect_nc_name=NC_FILE_TRANSECT,
        metrics_json_name=JSON_FILE_METRICS,
        config_name="config.yaml",
    )
    ds_diags = ds_diags.pipe(insert_dataset_r2).pipe(mse_to_rmse)
    config.pop("mapping_kwargs", None)  # this item clutters the report
    if args.commit_sha:
        config["commit"] = args.commit_sha

    report_sections: MutableMapping[str, Sequence[str]] = {}

    # histogram of timesteps used for testing
    try:
        timesteps = ds_diurnal["time"]
    except KeyError:
        pass
    else:
        timesteps = np.vectorize(vcm.cast_to_datetime)(timesteps)
        fig = fv3viz.plot_daily_and_hourly_hist(timesteps)
        fig.set_size_inches(10, 3)
        insert_report_figure(
            report_sections,
            fig,
            filename="timesteps_used.png",
            section_name="Timesteps used for testing",
            output_dir=temp_output_dir.name,
        )

    # Zonal average of vertical profiles for bias and R2
    zonal_avg_pressure_level_metrics = [
        var
        for var in ds_diags.data_vars
        if var.startswith("zonal_avg_pressure")
        and var.endswith("predict_vs_target")
        and ("r2" in var or "bias" in var)
    ]
    for var in sorted(zonal_avg_pressure_level_metrics):
        vmin, vmax = (0, 1) if "r2" in var.lower() else (None, None)
        fig = diagplot.plot_zonal_average(
            data=ds_diags[var],
            title=tidy_title(var),
            plot_kwargs={"vmin": vmin, "vmax": vmax},
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"zonal_avg_pressure_{var}.png",
            section_name="Zonal averaged pressure level metrics",
            output_dir=temp_output_dir.name,
        )

    # vertical profiles of bias and R2
    pressure_level_metrics = [
        var
        for var in ds_diags.data_vars
        if var.startswith("pressure_level") and var.endswith("predict_vs_target")
    ]
    for var in sorted(pressure_level_metrics):
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
    profiles = [
        var for var in ds_diags.data_vars if "dQ" in var and "z" in ds_diags[var].dims
    ] + ["Q1", "Q2"]
    for var in sorted(profiles):
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

    column_integrated_metrics = column_integrated_metric_names(metrics)

    # time averaged column integrated quantity maps
    for var in column_integrated_metrics:
        fig = diagplot.plot_column_integrated_var(
            ds_diags, var, derivation_plot_coords=ds_diags[DERIVATION_DIM].values,
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Time averaged maps",
            output_dir=temp_output_dir.name,
        )

    # column integrated quantity diurnal cycles
    for tag, var_group in [
        ("Q1_components", ["column_integrated_dQ1", "column_integrated_Q1"]),
        ("Q2_components", ["column_integrated_dQ2", "column_integrated_Q2"]),
    ]:
        fig = diagplot.plot_diurnal_cycles(
            ds_diurnal,
            vars=var_group,
            derivation_plot_coords=ds_diurnal[DERIVATION_DIM].values,
        )
        insert_report_figure(
            report_sections,
            fig,
            filename=f"{tag}.png",
            section_name="Diurnal cycles of column integrated quantities",
            output_dir=temp_output_dir.name,
        )

    # transect of predicted fields at lon=0
    transect_time = ds_transect.time.item()
    for var in sorted(ds_transect.data_vars):
        fig = plot_transect(ds_transect[var])
        insert_report_figure(
            report_sections,
            fig,
            filename=f"transect_lon0_{var}.png",
            section_name=f"Transect snapshot at lon=0 deg, {transect_time}",
            output_dir=temp_output_dir.name,
        )

    # scalar metrics for RMSE and bias
    metrics_formatted = []
    metrics = insert_scalar_metrics_r2(metrics, column_integrated_metrics)
    for var in sorted(column_integrated_metrics):
        values = {
            "r2": get_metric_string(metrics, "r2", var),
            "bias": " ".join(
                [get_metric_string(metrics, "bias", var), units_from_Q_name(var)]
            ),
        }
        metrics_formatted.append((var.replace("_", " "), values))
    metrics_formatted = OrderedDict(metrics_formatted)  # type: ignore

    for png in copy_pngs_to_report(args.input_path, temp_output_dir.name):
        report_sections[png] = [png]

    write_report(
        temp_output_dir.name,
        "ML offline diagnostics",
        report_sections,
        metadata=config,
        report_metrics=metrics_formatted,
    )

    copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_diurnal.close()
    ds_transect.close()

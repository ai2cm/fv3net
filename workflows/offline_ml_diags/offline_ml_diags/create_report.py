import argparse
import os
import atexit
import logging
import sys
import tempfile
from typing import MutableMapping, Sequence, List

import fv3viz
import numpy as np
import report
import vcm
from vcm.cloud import get_fs
import diagnostics_utils.plot as diagplot
from ._helpers import (
    get_metric_string,
    open_diagnostics_outputs,
    copy_outputs,
    tidy_title,
    units_from_name,
    column_integrated_metric_names,
    insert_dataset_r2,
    insert_scalar_metrics_r2,
    mse_to_rmse,
    is_3d,
    drop_physics_vars,
    drop_temperature_humidity_tendencies_if_not_predicted,
)
from ._select import plot_transect


DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"

NC_FILE_DIAGS = "offline_diagnostics.nc"
NC_FILE_DIURNAL = "diurnal_cycle.nc"
NC_FILE_TRANSECT = "transect_lon0.nc"
JSON_FILE_METRICS = "scalar_metrics.json"

MODEL_SENSITIVITY_HTML = "model_sensitivity.html"
TIME_MEAN_MAPS_HTML = "time_mean_maps.html"


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


def render_model_sensitivity(figures_dir, output_dir) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    for png in copy_pngs_to_report(figures_dir, output_dir):
        report_sections[png] = [png]

    return report.create_html(
        sections=report_sections, title="Model sensitivity to inputs",
    )


def render_time_mean_maps(
    figures_dir, output_dir, ds_diags, column_integrated_vars
) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    # time averaged column integrated quantity maps
    for var in column_integrated_vars:
        ds_diags[f"error_in_{var}"] = (
            ds_diags.sel(derivation="predict")[var]
            - ds_diags.sel(derivation="target")[var]
        )
        fig = diagplot.plot_column_integrated_var(
            ds_diags, var, derivation_plot_coords=ds_diags[DERIVATION_DIM].values,
        )
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Time averaged maps",
            output_dir=output_dir,
        )
        fig_error = diagplot.plot_column_integrated_var(
            ds_diags,
            f"error_in_{var}",
            derivation_plot_coords=None,
            derivation_dim=None,
        )
        report.insert_report_figure(
            report_sections,
            fig_error,
            filename=f"error_in_{var}.png",
            section_name="Time averaged maps",
            output_dir=output_dir,
        )
    return report.create_html(sections=report_sections, title="Time mean maps",)


def render_index(metrics, ds_diags, ds_diurnal, ds_transect, output_dir) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    # Links
    report_sections["Links"] = [
        report.Link("Model input sensitivity", MODEL_SENSITIVITY_HTML),
        report.Link("Time mean maps", TIME_MEAN_MAPS_HTML),
    ]

    # histogram of timesteps used for testing
    try:
        timesteps = ds_diurnal["time"]
    except KeyError:
        pass
    else:
        timesteps = np.vectorize(vcm.cast_to_datetime)(timesteps)
        fig = fv3viz.plot_daily_and_hourly_hist(timesteps)
        fig.set_size_inches(10, 3)
        report.insert_report_figure(
            report_sections,
            fig,
            filename="timesteps_used.png",
            section_name="Timesteps used for testing",
            output_dir=output_dir,
        )
    column_integrated_variable_names = column_integrated_metric_names(metrics)

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
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"zonal_avg_pressure_{var}.png",
            section_name="Zonal averaged pressure level metrics",
            output_dir=output_dir,
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
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Pressure level metrics",
            output_dir=output_dir,
        )

    # time averaged quantity vertical profiles over land/sea, pos/neg net precip
    profiles = [
        var
        for var in ds_diags.data_vars
        if ("Q1" in var or "Q2" in var) and is_3d(ds_diags[var])
    ]
    for var in sorted(profiles):
        fig = diagplot.plot_profile_var(
            ds_diags, var, derivation_dim=DERIVATION_DIM, domain_dim=DOMAIN_DIM,
        )
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Vertical profiles of predicted variables",
            output_dir=output_dir,
        )

    # 2d quantity diurnal cycles
    for var in ds_diurnal:
        fig = diagplot.plot_diurnal_cycles(
            ds_diurnal,
            var=var,
            derivation_plot_coords=ds_diurnal[DERIVATION_DIM].values,
        )
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Diurnal cycles of column integrated quantities",
            output_dir=output_dir,
        )

    # transect of predicted fields at lon=0
    if len(ds_transect) > 0:
        transect_time = ds_transect.time.item()
        for var in sorted(ds_transect.data_vars):
            fig = plot_transect(ds_transect[var])
            report.insert_report_figure(
                report_sections,
                fig,
                filename=f"transect_lon0_{var}.png",
                section_name=f"Transect snapshot at lon=0 deg, {transect_time}",
                output_dir=output_dir,
            )

    # scalar metrics for RMSE and bias
    metrics_formatted = []
    metrics = insert_scalar_metrics_r2(metrics, column_integrated_variable_names)
    for var in sorted(column_integrated_variable_names):
        values = {
            "r2": get_metric_string(metrics, "r2", var),
            "bias": " ".join(
                [get_metric_string(metrics, "bias", var), units_from_name(var)]
            ),
        }
        metrics_formatted.append((var.replace("_", " "), values))

    return report.create_html(
        sections=report_sections,
        title="ML offline diagnostics",
        metadata=config,
        metrics=dict(metrics_formatted),
    )


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
        config_name="data_config.yaml",
    )
    ds_diags = ds_diags.pipe(insert_dataset_r2).pipe(mse_to_rmse)

    # omit physics tendencies from report plots
    ds_diags = drop_physics_vars(ds_diags)
    ds_diurnal = drop_physics_vars(ds_diurnal)

    # diagnostics_utils currently fill dQ1/2 with zeros if not predicted
    # exclude these from the report if they are not model outputs.
    ds_diags = drop_temperature_humidity_tendencies_if_not_predicted(
        ds_diags, config["output_variables"]
    )
    ds_diurnal = drop_temperature_humidity_tendencies_if_not_predicted(
        ds_diurnal, config["output_variables"]
    )

    config.pop("mapping_kwargs", None)  # this item clutters the report
    if args.commit_sha:
        config["commit"] = args.commit_sha

    html_index = render_index(
        metrics, ds_diags, ds_diurnal, ds_transect, output_dir=temp_output_dir.name
    )
    with open(os.path.join(temp_output_dir.name, "index.html"), "w") as f:
        f.write(html_index)

    html_model_sensitivity = render_model_sensitivity(
        os.path.join(args.input_path, "model_sensitivity_figures"),
        output_dir=temp_output_dir.name,
    )
    with open(os.path.join(temp_output_dir.name, MODEL_SENSITIVITY_HTML), "w") as f:
        f.write(html_model_sensitivity)

    html_time_mean_maps = render_time_mean_maps(
        os.path.join(args.input_path, "time_mean_maps_figures"),
        temp_output_dir.name,
        ds_diags,
        column_integrated_vars=column_integrated_metric_names(metrics),
    )
    with open(os.path.join(temp_output_dir.name, TIME_MEAN_MAPS_HTML), "w") as f:
        f.write(html_time_mean_maps)

    copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_diurnal.close()
    ds_transect.close()

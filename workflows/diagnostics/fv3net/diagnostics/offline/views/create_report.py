import argparse
import os
import atexit
import logging
import sys
import tempfile
from typing import MutableMapping, Sequence, List
import fsspec

import fv3viz
import numpy as np
import report
import vcm
from vcm.cloud import get_fs
import yaml
from fv3net.diagnostics.offline._helpers import (
    get_metric_string,
    open_diagnostics_outputs,
    copy_outputs,
    tidy_title,
    units_from_name,
    is_3d,
)
from fv3net.diagnostics.offline._select import plot_transect
from fv3net.diagnostics.offline.compute import (
    DIAGS_NC_NAME,
    TRANSECT_NC_NAME,
    METRICS_JSON_NAME,
    METADATA_JSON_NAME,
    DERIVATION_DIM_NAME,
)
from .plot import (
    get_plot_dataset,
    plot_histogram,
    plot_profile_var,
    plot_diurnal_cycles,
    plot_zonal_average,
    plot_column_integrated_var,
    plot_generic_data_array,
    plot_histogram2d,
)
from fv3net.diagnostics._shared.constants import WVP, COL_DRYING

DOMAIN_DIM = "domain"

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
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help=("Training configuration yaml file to insert into report"),
    )
    parser.add_argument(
        "--training-data-config",
        type=str,
        default=None,
        help=("Training data configuration yaml file to insert into report"),
    )
    return parser.parse_args()


def render_model_sensitivity(figures_dir, output_dir) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    for png in copy_pngs_to_report(figures_dir, output_dir):
        report_sections[png] = [png]

    return report.create_html(
        sections=report_sections, title="Model sensitivity to inputs",
    )


def render_time_mean_maps(output_dir, ds_diags) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    ds_time_mean = get_plot_dataset(
        ds_diags, var_filter="time_mean", column_filters=["global"]
    )

    # snapshot maps
    snapshot_vars = [
        v
        for v in ds_diags
        if v.endswith("snapshot") and DERIVATION_DIM_NAME in ds_diags[v].dims
    ]
    for section, ds in zip(
        ["Time averaged maps", "Snapshot maps"], [ds_time_mean, ds_diags[snapshot_vars]]
    ):
        map_vars = [v for v in ds if not is_3d(ds[v])]
        for var in ds[map_vars]:
            ds[f"error_in_{var}"] = (
                ds.sel(derivation="predict")[var] - ds.sel(derivation="target")[var]
            )
            fig = plot_column_integrated_var(
                ds.update(ds_diags[["lat", "lon", "latb", "lonb"]]),
                var,
                derivation_plot_coords=ds_diags[DERIVATION_DIM_NAME].values,
            )
            report.insert_report_figure(
                report_sections,
                fig,
                filename=f"{var}.png",
                section_name=section,
                output_dir=output_dir,
            )
            fig_error = plot_column_integrated_var(
                ds.update(ds_diags[["lat", "lon", "latb", "lonb"]]),
                f"error_in_{var}",
                derivation_plot_coords=None,
                derivation_dim=None,
            )
            report.insert_report_figure(
                report_sections,
                fig_error,
                filename=f"error_in_{var}.png",
                section_name=section,
                output_dir=output_dir,
            )
    return report.create_html(sections=report_sections, title="Maps",)


def render_index(config, metrics, ds_diags, ds_transect, output_dir) -> str:
    report_sections: MutableMapping[str, Sequence[str]] = {}

    # Links
    report_sections["Links"] = [
        report.Link("Model input sensitivity", MODEL_SENSITIVITY_HTML),
        report.Link("Maps", TIME_MEAN_MAPS_HTML),
    ]

    # histogram of timesteps used for testing
    try:
        timesteps = ds_diags["time"]
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

    # Zonal average of vertical profiles for R2
    zonal_avg_pressure_level_metrics = [
        var
        for var in ds_diags.data_vars
        if var.endswith("_pressure_level_zonal_avg_global") and ("r2" in var.lower())
    ]
    for var in sorted(zonal_avg_pressure_level_metrics):
        fig = plot_zonal_average(
            data=ds_diags[var],
            title=tidy_title(var),
            plot_kwargs={"vmin": 0, "vmax": 1},
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
        if var.endswith("pressure_level_global") and ("r2" in var.lower())
    ]
    for var in sorted(pressure_level_metrics):
        ylim = (0, 1) if "r2" in var.lower() else None
        fig = plot_generic_data_array(
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
    vars_3d = [v for v in ds_diags if is_3d(ds_diags[v])]
    ds_profiles = get_plot_dataset(
        ds_diags[vars_3d],
        var_filter="time_domain_mean",
        column_filters=[
            "global",
            "land",
            "sea",
            "positive_net_precipitation",
            "negative_net_precipitation",
        ],
    )
    for var in sorted(ds_profiles):
        fig = plot_profile_var(
            ds_profiles, var, derivation_dim=DERIVATION_DIM_NAME, domain_dim="domain",
        )
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"{var}.png",
            section_name="Vertical profiles of predicted variables",
            output_dir=output_dir,
        )

    # 2d quantity diurnal cycles
    ds_diurnal = get_plot_dataset(
        ds_diags, var_filter="diurnal_cycle", column_filters=["global", "land", "sea"]
    )
    for var in ds_diurnal:
        fig = plot_diurnal_cycles(ds_diurnal, var=var,)
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
    scalar_vars_r2 = sorted([var for var in metrics if "r2" in var])
    scalar_vars_bias = [var.replace("_r2", "_bias") for var in scalar_vars_r2]

    for var_r2, var_bias in zip(scalar_vars_r2, scalar_vars_bias):
        values = {
            "r2": get_metric_string(metrics[var_r2]),
            "bias": " ".join(
                [get_metric_string(metrics[var_bias]), units_from_name(var_bias)]
            ),
        }
        metrics_formatted.append((var_r2.replace("_r2", ""), values))

    if "water_vapor_path_versus_minus_column_integrated_q2_hist_2d" in ds_diags:
        hist2d_wvp_vs_q2 = plot_histogram2d(
            ds_diags, "water_vapor_path", "minus_column_integrated_q2"
        )
        report.insert_report_figure(
            report_sections,
            hist2d_wvp_vs_q2,
            filename=f"hist2d_wvp_vs_q2.png",
            section_name=f"Water vapor path versus column integrated drying",
            output_dir=output_dir,
        )
        hist_wvp = plot_histogram(ds_diags, f"{WVP}_histogram")
        report.insert_report_figure(
            report_sections,
            hist_wvp,
            filename=f"hist_wvp.png",
            section_name=f"Water vapor path versus column integrated drying",
            output_dir=output_dir,
        )
        hist_col_drying = plot_histogram(ds_diags, f"{COL_DRYING}_histogram")
        report.insert_report_figure(
            report_sections,
            hist_col_drying,
            filename=f"hist_col_drying.png",
            section_name=f"Water vapor path versus column integrated drying",
            output_dir=output_dir,
        )
    return report.create_html(
        sections=report_sections,
        title="ML offline diagnostics",
        metadata=config,
        metrics=dict(metrics_formatted),
    )


def create_report(args):
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    ds_diags, ds_transect, metrics, metadata = open_diagnostics_outputs(
        args.input_path,
        diagnostics_nc_name=DIAGS_NC_NAME,
        transect_nc_name=TRANSECT_NC_NAME,
        metrics_json_name=METRICS_JSON_NAME,
        metadata_json_name=METADATA_JSON_NAME,
    )

    if args.commit_sha:
        metadata["commit"] = args.commit_sha
    if args.training_config:
        with fsspec.open(args.training_config, "r") as f:
            metadata["training_config"] = yaml.safe_load(f)
    if args.training_data_config:
        with fsspec.open(args.training_data_config, "r") as f:
            metadata["training_data_config"] = yaml.safe_load(f)

    html_index = render_index(
        metadata, metrics, ds_diags, ds_transect, output_dir=temp_output_dir.name,
    )
    with open(os.path.join(temp_output_dir.name, "index.html"), "w") as f:
        f.write(html_index)

    html_model_sensitivity = render_model_sensitivity(
        os.path.join(args.input_path, "model_sensitivity_figures"),
        output_dir=temp_output_dir.name,
    )
    with open(os.path.join(temp_output_dir.name, MODEL_SENSITIVITY_HTML), "w") as f:
        f.write(html_model_sensitivity)

    html_time_mean_maps = render_time_mean_maps(temp_output_dir.name, ds_diags,)
    with open(os.path.join(temp_output_dir.name, TIME_MEAN_MAPS_HTML), "w") as f:
        f.write(html_time_mean_maps)

    copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # Explicitly call .close() or xarray raises errors atexit
    # described in https://github.com/shoyer/h5netcdf/issues/50
    ds_diags.close()
    ds_transect.close()


if __name__ == "__main__":
    logger.info("Starting create report routine.")
    args = _create_arg_parser()
    create_report(args)

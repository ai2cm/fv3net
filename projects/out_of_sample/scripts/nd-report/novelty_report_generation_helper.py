import fv3viz

import argparse
import dataclasses
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3net.diagnostics.offline._helpers import copy_outputs
import matplotlib.pyplot as plt
import numpy as np
import os
import report
from report.create_report import Metrics
import tempfile
from typing import List, MutableMapping, Sequence
import vcm
from vcm.catalog import catalog
import xarray as xr


@dataclasses.dataclass
class OOSModel:
    name: str
    nd: NoveltyDetector
    nd_path: str
    cutoff: float


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", default=None, type=str, help=("Path to yaml config file."),
    )
    parser.add_argument(
        "--n_weeks", default=None, type=int, help=("Compute on the first n_weeks"),
    )
    parser.add_argument(
        "--time_sample_freq",
        default=None,
        type=str,
        help=("Resampling frequency, e.g. '12H'."),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        type=str,
        help=("List of variable names to load"),
    )
    return parser


def make_diagnostic_plots(
    report_sections: MutableMapping[str, Sequence[str]],
    temp_output_dir: str,
    models: List[OOSModel],
    model_diags: List[xr.Dataset],
    grid: xr.Dataset,
    has_cutoff_plots: bool = False,
) -> Metrics:
    """
    Fills in sections of a future html report for a collection of models with
    diagnostic data. The first section is a time plot that compares the fraction
    of novelties over time for each model on the target dataset. The remaining
    sections are for each model, which includes 4 plots: (1) a map plot of the
    average rate of novelties over time, (2) a Hovmoller plot of the novelty fraction
    by time and latitude, (3) the fraction of novelties for each as a function
    of the chosen cutoff, and (4) a histogram of novelty scores.
    """
    # time plot for all model novelty fractions
    fig = plt.figure()
    for model in models:
        diags = model_diags[model.name]
        xr.where(diags.centered_score > model.cutoff, 1, 0).mean(
            ["tile", "x", "y"]
        ).plot(label=model.name)
    plt.legend()
    plt.title("Fraction of novelties over time")
    report.insert_report_figure(
        report_sections,
        fig,
        filename=f"novelties-over-time.png",
        section_name=f"Novelty frequency over time",
        output_dir=temp_output_dir,
    )
    plt.close(fig)

    metrics = {"fraction novelties": {}}

    for model in models:
        diags = model_diags[model.name]
        is_novelty = xr.where(diags.centered_score > model.cutoff, 1, 0)

        # (1) map novelty plot
        fig, _, _, _, _ = fv3viz.plot_cube(
            ds=is_novelty.mean("time").to_dataset(name="novelty_frac").merge(grid),
            var_name="novelty_frac",
            vmax=1,
            vmin=0,
        )
        plt.title(model.name)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"time-avg-{model.name}.png",
            section_name="Time-averaged novelty frequency",
            output_dir=temp_output_dir,
        )
        plt.close(fig)

        # (2) hovmoller novelty plot
        fig = plt.figure()
        vcm.zonal_average_approximate(grid.lat, is_novelty).plot(x="time")
        plt.title(model.name)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"hovmoller-{model.name}.png",
            section_name="Hovmoller plots of novelty frequency",
            output_dir=temp_output_dir,
        )
        plt.close(fig)

        # (3) fraction of novelties per cutoff (optional)
        if has_cutoff_plots:
            fig = plt.figure()
            min_score = diags.centered_score.min().values
            max_score = diags.centered_score.max().values
            cutoffs = np.linspace(min_score, max_score, 20)
            frac_novelty = [
                xr.where(diags.centered_score > c, 1, 0).mean().values for c in cutoffs
            ]
            plt.plot(cutoffs, frac_novelty)
            plt.title(model.name)
            plt.axvline(x=model.cutoff)
            report.insert_report_figure(
                report_sections,
                fig,
                filename=f"cutoff-{model.name}.png",
                section_name="Fraction of novelties at varied score cutoffs",
                output_dir=temp_output_dir,
            )
            plt.close(fig)

        # (4) histogram of scores
        fig = plt.figure()
        diags.centered_score.plot(bins=50)
        plt.title(model.name)
        plt.axvline(x=model.cutoff)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"histogram-{model.name}.png",
            section_name="Histograms of novelty scores",
            output_dir=temp_output_dir,
        )
        plt.close(fig)

        # (5) total fraction of novelties for metrics table
        metrics["fraction novelties"][model.name] = is_novelty.mean().values

    return metrics


def generate_and_save_report(
    models, model_diags, has_cutoff_plots, metadata, report_url, title
):
    common_coords = {"tile": range(6), "x": range(48), "y": range(48)}
    grid = catalog["grid/c48"].read().assign_coords(common_coords)
    report_sections: MutableMapping[str, Sequence[str]] = {}
    with tempfile.TemporaryDirectory() as temp_output_dir:
        metrics = make_diagnostic_plots(
            report_sections,
            temp_output_dir,
            models,
            model_diags,
            grid,
            has_cutoff_plots,
        )

        # creates html text based on the above sections
        html_index = report.create_html(
            sections=report_sections,
            title=title,
            metadata=metadata,
            metrics=metrics,
            collapse_metadata=True,
        )

        with open(os.path.join(temp_output_dir, "index.html"), "w") as f:
            f.write(html_index)

        copy_outputs(temp_output_dir, report_url)
        print(f"Saved report to {report_url}")
        if report_url.startswith("gs://"):
            web_url = report_url.replace("gs://", "https://storage.googleapis.com/")
            web_url = os.path.join(web_url, "index.html")
            print(f"Find this report now at {web_url}.")

import argparse
import atexit
import dataclasses
import fsspec
import fv3fit
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3net.diagnostics.offline._helpers import copy_outputs
import fv3viz
import hashlib
from lark import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import report
import tempfile
from typing import List, MutableMapping, Sequence
import uuid
import vcm
from vcm.catalog import catalog
import xarray as xr
import yaml


@dataclasses.dataclass
class OOSModel:
    name: str
    nd: NoveltyDetector
    nd_path: str

def _cleanup_temp_dir(temp_dir):
    logger.info(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        default=None,
        type=str,
        help=("Path to yaml config file."),
    )
    return parser

_STATE_SUFFIX = "state_after_timestep.zarr"

def get_diags_offline_suffix(model_name: str) -> str:
    return f"diags_novelty_offline_{model_name}.zarr"


def get_diags(oos_model: OOSModel, ds_path: str) -> xr.Dataset:
    """
    Returns a dataset containing the is_novelty and novelty_score fields that reflect
    the offline behavior of a novelty detector on some other temporal dataset.
    """
    diags_url = os.path.join(
        oos_model.nd_path,
        f"diags_novelty_offline/{hashlib.md5(ds_path.encode()).hexdigest()}"
    )
    fs = vcm.cloud.get_fs(diags_url)
    if fs.exists(diags_url):
        print(
            f"Reading offline novelty data from "
            + f"previous computation, at {diags_url}."
        )
        diags = xr.open_zarr(fsspec.get_mapper(diags_url))
    else:
        state_url = os.path.join(ds_path, _STATE_SUFFIX)
        print(f"Computing offline novelty data from states at {state_url}.")
        ds = xr.open_zarr(fsspec.get_mapper(state_url))
        _, diags = oos_model.nd.predict_novelties(ds)
        mapper = fsspec.get_mapper(diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {diags_url}.")
    return diags


def make_diagnostic_plots(
    report_sections: MutableMapping[str, Sequence[str]],
    temp_output_dir: str,
    models: List[OOSModel],
    model_diags: List[xr.Dataset],
    grid: xr.Dataset
):
    """
    Fills in sections of a future html report for a collection of models with
    diagnostic data. The first section is a time plot that compares the fraction
    of novelties over time for each model on the target dataset. The remaining
    sections are for each model, which includes 4 plots: (1) a map plot of the
    average rate of novelties over time, (2) a Hovmoller plot of the novelty fraction
    by time and latitude, (3) the fraction of novelties for each as a function
    of the chosen cutoff, and (4) a histogram of novelty scores.
    """
    fig = plt.figure()
    for model in models:
        diags = model_diags[model.name]
        diags.is_novelty.mean(["tile", "x", "y"]).plot(label=model.name)
    plt.legend()
    plt.title("Fraction of novelties over time")
    report.insert_report_figure(
        report_sections,
        fig,
        filename=f"novelties-over-time.png",
        section_name=f"Novelty frequency over time",
        output_dir=temp_output_dir.name,
    )
    plt.close(fig)

    for model in models:
        diags = model_diags[model.name]

        fig, _, _, _, _ = fv3viz.plot_cube(
            ds=diags.is_novelty.mean("time")
            .to_dataset(name="novelty_frac")
            .merge(grid),
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
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        vcm.zonal_average_approximate(grid.lat, diags.is_novelty).plot(x="time")
        plt.title(model.name)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"hovmoller-{model.name}.png",
            section_name="Hovmoller plots of novelty frequency",
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        min_score = diags.novelty_score.min().values
        max_score = diags.novelty_score.max().values
        cutoffs = np.linspace(min_score, max_score, 20)
        frac_novelty = [
            xr.where(diags.novelty_score > cutoff, 1, 0).mean().values
            for cutoff in cutoffs
        ]
        plt.plot(cutoffs, frac_novelty)
        plt.title(model.name)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"cutoff-{model.name}.png",
            section_name="Fraction of novelties at varied score cutoffs",
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        diags.novelty_score.plot(bins=50)
        plt.title(model.name)
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"histogram-{model.name}.png",
            section_name="Histograms of novelty scores",
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)


def create_report(args):
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    common_coords = {"tile": range(6), "x": range(48), "y": range(48)}
    grid = catalog["grid/c48"].read().assign_coords(common_coords)

    with open(args.config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    models = [
        OOSModel(
            model["name"],
            fv3fit.load(model["model_url"]),
            model["model_url"]
        ) for model in config["models"]
    ]
    run_name = config["run_dataset"]["name"]
    run_url = config["run_dataset"]["url"]
    report_url = config["report_url"]
    report_id = uuid.uuid4().hex
    if config["append_random_id_to_url"]:
        report_url = os.path.join(report_url, f"report-{report_id}")

    metadata = {
        "models": [{
            "name": model.name,
            "model_url": model.nd_path,
        } for model in models],
        "run_name": run_name,
        "run_url": run_url,
        "report_url": report_url
    }
    report_sections: MutableMapping[str, Sequence[str]] = {}

    model_diags = {}
    for model in models:
        model_diags[model.name] = get_diags(model, run_url)
        
    make_diagnostic_plots(
        report_sections,
        temp_output_dir,
        models,
        model_diags,
        grid
    )

    # creates html text based on the above sections
    html_index = report.create_html(
        sections=report_sections,
        title=f"Offline metrics for out-of-sample analysis on {run_name}",
        metadata=metadata,
        metrics=None,
        collapse_metadata=True,
    )

    with open(os.path.join(temp_output_dir.name, "index.html"), "w") as f:
        f.write(html_index)

    copy_outputs(temp_output_dir.name, report_url)
    print(f"Saved report to {report_url}")
    if report_url.startswith("gs://"):
        web_url = report_url.replace("gs://", "https://storage.googleapis.com/")
        web_url = os.path.join(web_url, "index.html")
        print(f"Find this report now at {web_url}.")


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    create_report(args)

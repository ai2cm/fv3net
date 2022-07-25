import argparse
import atexit
import dataclasses
import random
import string
import fsspec
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit.sklearn._min_max_novelty_detector import MinMaxNoveltyDetector
from fv3fit.sklearn._ocsvm_novelty_detector import OCSVMNoveltyDetector
from fv3net.diagnostics.offline._helpers import copy_outputs
import fv3viz
from lark import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import report
import tempfile
from typing import MutableMapping, Sequence
import vcm
from vcm.catalog import catalog
import xarray as xr


@dataclasses.dataclass
class OOSModel:
    name: str
    nd: NoveltyDetector
    ds_path: str


def _cleanup_temp_dir(temp_dir):
    logger.info(f"Cleaning up temp dir {temp_dir.name}")
    temp_dir.cleanup()


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    return parser


_DIAGS_SUFFIX = "diags.zarr"
_NOVELTY_DIAGS_SUFFIX = "diags_novelty.zarr"
_STATE_SUFFIX = "state_after_timestep.zarr"


def get_diags_offline_suffix(model_name: str) -> str:
    return f"diags_novelty_offline_{model_name}.zarr"


def get_online_diags(oos_model: OOSModel) -> xr.Dataset:
    """
    Returns a dataset containing the is_novelty and novelty_score fields that reflect
    how a given novelty detector behaved on its online run.
    """
    diags_url = os.path.join(oos_model.ds_path, _DIAGS_SUFFIX)
    diags = xr.open_zarr(fsspec.get_mapper(diags_url))
    fsspec.get_mapper
    novelty_diags_url = os.path.join(oos_model.ds_path, _NOVELTY_DIAGS_SUFFIX)
    fs = vcm.cloud.get_fs(novelty_diags_url)
    if "is_novelty" in diags.data_vars:
        print(f"Reading online novelty data from model diagnostics, at {diags_url}.")
        return diags
    elif fs.exists(novelty_diags_url):
        print(
            f"Reading online novelty data from "
            + f"previous computation, at {novelty_diags_url}."
        )
        diags = xr.open_zarr(fsspec.get_mapper(novelty_diags_url))
        return diags
    else:
        state_url = os.path.join(oos_model.ds_path, _STATE_SUFFIX)
        print(f"Computing online novelty data from states at {state_url}.")
        ds = xr.open_zarr(fsspec.get_mapper(state_url))
        _, diags = oos_model.nd.predict_novelties(ds)
        mapper = fsspec.get_mapper(novelty_diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {novelty_diags_url}.")
        return diags


def get_diags(oos_model: OOSModel, ds_path) -> xr.Dataset:
    """
    Returns a dataset containing the is_novelty and novelty_score fields that reflect
    the offline behavior of a novelty detector on some other temporal dataset.
    """
    diags_url = os.path.join(ds_path, get_diags_offline_suffix(oos_model.name))
    fs = vcm.cloud.get_fs(diags_url)
    if fs.exists(diags_url):
        print(
            f"Reading offline novelty data from "
            + f"previous computation, at {diags_url}."
        )
        diags = xr.open_zarr(fsspec.get_mapper(diags_url))
    else:
        state_url = os.path.join(oos_model.ds_path, _STATE_SUFFIX)
        print(f"Computing offline novelty data from states at {state_url}.")
        ds = xr.open_zarr(fsspec.get_mapper(state_url))
        _, diags = oos_model.nd.predict_novelties(ds)
        mapper = fsspec.get_mapper(diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {diags_url}.")
    return diags


def make_diagnostic_plots(
    report_sections, temp_output_dir, models, model_diags, grid, type
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
    type_file_name = type.lower().replace(" ", "-")
    fig = plt.figure()
    for model in models:
        diags = model_diags[model.name]
        diags.is_novelty.mean(["tile", "x", "y"]).plot(label=model.name)
    plt.legend()
    plt.title("Fraction of novelties over time")
    report.insert_report_figure(
        report_sections,
        fig,
        filename=f"novelties-over-time-{type_file_name}.png",
        section_name=f"{type} novelty frequency over time",
        output_dir=temp_output_dir.name,
    )
    plt.close(fig)

    for model in models:
        section_name = f"{type} novelty frequency of {model.name}"
        diags = model_diags[model.name]

        print(model.name)

        fig, _, _, _, _ = fv3viz.plot_cube(
            ds=diags.is_novelty.mean("time")
            .to_dataset(name="novelty_frac")
            .merge(grid),
            var_name="novelty_frac",
            vmax=1,
            vmin=0,
        )
        plt.title("Time-averaged fraction of novelties")
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"time-avg-{model.name}-{type_file_name}.png",
            section_name=section_name,
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        vcm.zonal_average_approximate(grid.lat, diags.is_novelty).plot(x="time")
        plt.title("Hovmoller plot of fraction of novelties")
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"hovmoller-{model.name}-{type_file_name}.png",
            section_name=section_name,
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        if isinstance(model.nd, OCSVMNoveltyDetector):
            min_score = diags.novelty_score.min().values
            cutoffs = np.linspace(min_score, 0, 20)
        elif isinstance(model.nd, MinMaxNoveltyDetector):
            max_score = diags.novelty_score.max().values
            cutoffs = np.linspace(0, max_score, 20)
        frac_novelty = [
            xr.where(diags.novelty_score > cutoff, 1, 0).mean().values
            for cutoff in cutoffs
        ]
        plt.plot(cutoffs, frac_novelty)
        plt.title("Fraction of novelties for given cutoff")
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"cutoff-{model.name}-{type_file_name}.png",
            section_name=section_name,
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

        fig = plt.figure()
        diags.novelty_score.plot(bins=50)
        plt.title("Histogram of novelty scores")
        report.insert_report_figure(
            report_sections,
            fig,
            filename=f"histogram-{model.name}-{type_file_name}.png",
            section_name=section_name,
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)


def create_report(args):
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    # if there's no output path, generates a new one
    id = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
    )
    default_output_path = f"gs://vcm-ml-public/claytons/oos_analysis/report_{id}"
    output_path = args.output_path or default_output_path

    common_coords = {"tile": range(6), "x": range(48), "y": range(48)}
    grid = catalog["grid/c48"].read().assign_coords(common_coords)

    # the novelty detectors and the prognostic runs they induced
    minmax_model = OOSModel(
        "minmax",
        MinMaxNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-13/oos-test/novelty"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-13/oos-trial2/fv3gfs_run",
    )
    ocsvm_default_model = OOSModel(
        "ocsvm-1/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-15/"
            + "ocsvm-default-trial-3/trained_models/ocsvm"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-16/"
        + "ocsvm-test-default-trial7/fv3gfs_run",
    )
    ocsvm_large_model = OOSModel(
        "ocsvm-4/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-18/"
            + "ocsvm-gamma-trial-1/trained_models/ocsvm-large"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-19/"
        + "oos-large-gamma-trial6/fv3gfs_run",
    )
    ocsvm_small_model = OOSModel(
        "ocsvm-1/4/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-18/"
            + "ocsvm-gamma-trial-1/trained_models/ocsvm-small"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-19/"
        + "oos-small-gamma-trial3/fv3gfs_run",
    )
    models = [minmax_model, ocsvm_small_model, ocsvm_default_model, ocsvm_large_model]

    # additional datasets to evaluate the models on
    baseline_dataset_url = (
        "gs://vcm-ml-experiments/claytons/2022-07-19/baseline-trial2/fv3gfs_run"
    )
    prognostic_dataset_url = (
        "gs://vcm-ml-experiments/claytons/2022-07-14/oos-compare-trial1/fv3gfs_run"
    )

    report_sections: MutableMapping[str, Sequence[str]] = {}

    # online diagnostics: novelty detector evaluated the data generated by the model
    model_diags = {}
    for model in models:
        model_diags[model.name] = get_online_diags(model)
    make_diagnostic_plots(
        report_sections, temp_output_dir, models, model_diags, grid, "Online"
    )

    # offline diagnostics: novelty detectors are evaluated on other fixed datasets:
    # the ML-nudged prognostic run and the no-ML baseline
    for (name, url) in zip(
        ["baseline", "prognostic"], [baseline_dataset_url, prognostic_dataset_url]
    ):
        model_diags = {}
        for model in models:
            model_diags[model.name] = get_diags(model, url)
        make_diagnostic_plots(
            report_sections,
            temp_output_dir,
            models,
            model_diags,
            grid,
            f"Offline {name}",
        )

    # creates html text based on the above sections
    html_index = report.create_html(
        sections=report_sections,
        title="Offline metrics for out-of-sample analysis",
        metadata=None,
        metrics=None,
        collapse_metadata=True,
    )

    with open(os.path.join(temp_output_dir.name, "index.html"), "w") as f:
        f.write(html_index)

    copy_outputs(temp_output_dir.name, output_path)
    print(f"Save report to {output_path}")


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    create_report(args)

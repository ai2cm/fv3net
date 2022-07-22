import argparse
import atexit
import dataclasses
import fsspec
from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit.sklearn._min_max_novelty_detector import MinMaxNoveltyDetector
from fv3fit.sklearn._ocsvm_novelty_detector import OCSVMNoveltyDetector
from fv3net.artifacts.metadata import StepMetadata
from fv3net.diagnostics.offline._helpers import copy_outputs
import json
import fv3viz
from lark import logger
import matplotlib.pyplot as plt
import os
import report
import sys
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
    # parser.add_argument(
    #     "input_path", type=str, help=("Location of diagnostics and metrics data."),
    # )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    # parser.add_argument(
    #     "--commit-sha",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Commit SHA of fv3net used to create report. Useful for referencing"
    #         "the version used to train the model."
    #     ),
    # )
    # parser.add_argument(
    #     "--training-config",
    #     type=str,
    #     default=None,
    #     help=("Training configuration yaml file to insert into report"),
    # )
    # parser.add_argument(
    #     "--training-data-config",
    #     type=str,
    #     default=None,
    #     help=("Training data configuration yaml file to insert into report"),
    # )
    # parser.add_argument(
    #     "--no-wandb",
    #     help=(
    #         "Disable logging of run to wandb. Uses
    #
    #
    # environment variables WANDB_ENTITY, "
    #         "WANDB_PROJECT, WANDB_JOB_TYPE as wandb.init options."
    #     ),
    #     action="store_true",
    # )
    return parser


def get_online_diags(oosModel: OOSModel) -> xr.Dataset:
    diags = xr.open_zarr(
        fsspec.get_mapper(os.path.join(oosModel.ds_path, "diags.zarr"))
    )
    if "is_novelty" in diags.data_vars:
        return diags
    else:
        ds = xr.open_zarr(
            fsspec.get_mapper(
                os.path.join(oosModel.ds_path, "state_after_timestep.zarr/")
            )
        )
        print(ds)
        return oosModel.nd.predict(ds)


def get_diags(nd: NoveltyDetector, ds_path) -> xr.Dataset:
    ds = xr.open_zarr(
        fsspec.get_mapper(os.path.join(ds_path, "state_after_timestep.zarr/"))
    )
    return nd.predict(ds)


def create_report(args):
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    common_coords = {"tile": range(6), "x": range(48), "y": range(48)}
    grid = catalog["grid/c48"].read().assign_coords(common_coords)

    # TODO: handle arguments

    # TODO: import datasets
    metrics = {}
    # metrics_formatted = []
    # metadata = {}

    # prognostic_ds_path = (
    #     "gs://vcm-ml-experiments/claytons/2022-07-14/oos-compare-trial1/fv3gfs_run"
    # )
    minmaxModel = OOSModel(
        "minmax",
        MinMaxNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-13/oos-test/novelty"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-13/oos-trial2/fv3gfs_run",
    )
    ocsvmDefaultModel = OOSModel(
        "ocsvm-1/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-15/"
            + "ocsvm-default-trial-3/trained_models/ocsvm"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-16/"
        + "ocsvm-test-default-trial7/fv3gfs_run",
    )
    ocsvmLargeModel = OOSModel(
        "ocsvm-4/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-18/"
            + "ocsvm-gamma-trial-1/trained_models/ocsvm-large"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-19/"
        + "oos-large-gamma-trial6/fv3gfs_run",
    )
    ocsvmSmallModel = OOSModel(
        "ocsvm-1/4/79",
        OCSVMNoveltyDetector.load(
            "gs://vcm-ml-experiments/claytons/2022-07-18/"
            + "ocsvm-gamma-trial-1/trained_models/ocsvm-small"
        ),
        "gs://vcm-ml-experiments/claytons/2022-07-19/"
        + "oos-small-gamma-trial3/fv3gfs_run",
    )
    models = [minmaxModel, ocsvmSmallModel, ocsvmDefaultModel, ocsvmLargeModel]

    report_sections: MutableMapping[str, Sequence[str]] = {}

    for model in models:
        section_name = f"Novelty frequency of {model.name}"
        diags = get_online_diags(model)
        # diags = xr.open_zarr(fsspec.get_mapper(os.path.join(data_dir, 'diags.zarr')))

        # TODO: create plots from datasets

        # fig = plt.figure()
        # plt.plot([0, 1], [1, 0])
        # plt.title("title")
        # report.insert_report_figure(
        #     report_sections,
        #     fig,
        #     filename=f"fig.png",
        #     section_name="Section",
        #     output_dir=temp_output_dir.name,
        # )
        # plt.close(fig)

        print(model.name)
        print(diags)

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
            filename=f"time-avg-{model.name}.png",
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
            filename=f"hovmoller-{model.name}.png",
            section_name=section_name,
            output_dir=temp_output_dir.name,
        )
        plt.close(fig)

    html_index = report.create_html(
        sections=report_sections,
        title="Offline metrics for out-of-sample analysis",
        metadata=None,
        metrics=None,
        collapse_metadata=True,
    )

    with open(os.path.join(temp_output_dir.name, "index.html"), "w") as f:
        f.write(html_index)

    copy_outputs(temp_output_dir.name, args.output_path)
    logger.info(f"Save report to {args.output_path}")

    # Gcloud logging allows metrics to get ingested into database
    print(json.dumps({"json": metrics}))
    StepMetadata(
        job_type="offline_report",
        url=args.output_path,
        dependencies={
            # "offline_diagnostics": args.input_path,
            # "model": metadata.get("model_path"),
        },
        args=sys.argv[1:],
    ).print_json()


if __name__ == "__main__":
    logger.info("Starting create report routine.")
    parser = _get_parser()
    args = parser.parse_args()
    create_report(args)

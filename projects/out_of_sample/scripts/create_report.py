import argparse
import atexit
import json
from fv3net.artifacts.metadata import StepMetadata
from fv3net.diagnostics.offline._helpers import copy_outputs
from lark import logger
import matplotlib.pyplot as plt
import os
import report
import sys
import tempfile
from typing import MutableMapping, Sequence


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


def create_report(args):
    temp_output_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_output_dir)

    # TODO: import datasets
    metrics = {}
    # metrics_formatted = []
    # metadata = {}

    # TODO: handle arguments

    # TODO: create plots from datasets
    report_sections: MutableMapping[str, Sequence[str]] = {}

    fig = plt.figure()
    plt.plot([0, 1], [1, 0])
    plt.title("title")
    report.insert_report_figure(
        report_sections,
        fig,
        filename=f"fig.png",
        section_name="Section",
        output_dir=temp_output_dir.name,
    )
    plt.close(fig)

    html_index = report.create_html(
        sections=report_sections,
        title="This is a page",
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

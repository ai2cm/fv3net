from plots import make_all_plots
from config import (
    OUTPUT_NC_FILENAME,
    CONFIG_FILENAME,
    FIGURE_METADATA_FILE,
    METADATA_TABLE_FILE,
    REPORT_TITLE,
    INIT_TIME_DIM,
)
from vcm.cloud.gsutil import copy
from vcm.cloud import get_protocol
import report
import xarray as xr
import fsspec
import yaml
import argparse
import os
import shutil
import logging
import sys

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
out_hdlr.setLevel(logging.INFO)
logging.basicConfig(handlers=[out_hdlr], level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "netcdf_path",
        type=str,
        help="Location of diagnostics netcdf file output from diags workflow.",
    )
    parser.add_argument(
        "--report_directory",
        type=str,
        default=None,
        help="(Public) bucket path for report and image upload. If omitted, report is"
        "written to netcdf_path.",
    )

    return parser


def _write_report(output_report_dir, report_sections, metadata):

    with open(os.path.join(output_report_dir, FIGURE_METADATA_FILE), mode="w") as f:
        yaml.dump(report_sections, f)
    with open(os.path.join(output_report_dir, METADATA_TABLE_FILE), mode="w") as f:
        yaml.dump(metadata, f)
    filename = REPORT_TITLE.replace(" ", "_").replace("-", "_").lower() + ".html"
    html_report = report.create_html(report_sections, REPORT_TITLE, metadata=metadata)
    with open(os.path.join(output_report_dir, filename), "w") as f:
        f.write(html_report)


if __name__ == "__main__":

    args = _create_arg_parser().parse_args()

    with fsspec.open(
        os.path.join(args.netcdf_path, CONFIG_FILENAME), "r"
    ) as config_file:
        config = yaml.safe_load(config_file)

    output_nc_path = os.path.join(args.netcdf_path, OUTPUT_NC_FILENAME)
    with fsspec.open(output_nc_path) as ncfile:
        states_and_tendencies = xr.open_dataset(ncfile).load()

    # if report output path is GCS location, save results to local output dir first

    if args.report_directory:
        report_path = args.report_directory
    else:
        report_path = args.netcdf_path

    proto = get_protocol(report_path)
    if proto == "" or proto == "file":
        output_report_dir = report_path
    elif proto == "gs":
        remote_report_path, output_report_dir = os.path.split(report_path.strip("/"))

    if os.path.exists(output_report_dir):
        shutil.rmtree(output_report_dir)
    os.mkdir(output_report_dir)

    logger.info(f"Writing diagnostics plots report to {report_path}")

    report_sections = make_all_plots(states_and_tendencies, config, output_report_dir)
    metadata = vars(args)
    metadata.update(
        {"initializations_processed": states_and_tendencies.attrs[INIT_TIME_DIM]}
    )
    _write_report(output_report_dir, report_sections, metadata)

    # copy report directory to necessary locations
    if proto == "gs":
        copy(output_report_dir, remote_report_path)
    if args.report_directory:
        copy(output_report_dir, args.netcdf_path)

import argparse
import os
import tempfile
import xarray as xr
import yaml

import fsspec
from vcm.cloud import get_fs
from vcm.cloud.gsutil import copy
import vcm
from vcm import safe

import report

from ..data import merge_comparison_datasets
from .data import (
    predict_on_test_data,
    load_high_res_diag_dataset,
    add_column_heating_moistening,
)
from .diagnostics import plot_diagnostics
from .create_metrics import create_metrics_dataset
from .plot_metrics import plot_metrics
from .plot_timesteps import plot_timestep_counts
import logging

DATASET_NAME_PREDICTION = "prediction"
DATASET_NAME_FV3_TARGET = "C48_target"
DATASET_NAME_SHIELD_HIRES = "coarsened_high_res"
TIMESTEPS_USED_FILENAME = "timesteps_used.yml"
REPORT_TITLE = "ML offline diagnostics"

DPI_FIGURES = {
    "timestep_histogram": 90,
    "LTS": 100,
    "dQ2_pressure_profiles": 100,
    "R2_pressure_profiles": 100,
    "diurnal_cycle": 90,
    "map_plot_3col": 120,
    "map_plot_single": 100,
}


logger = logging.getLogger(__file__)


def _is_remote(path):
    return path.startswith("gs://")


def _write_report(output_dir, sections, metadata, title):
    filename = title.replace(" ", "_") + ".html"
    html_report = report.create_html(sections, title, metadata=metadata)
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(html_report)


def compute_metrics_and_plot(ds, output_dir, names):
    # TODO refactor this "and" function into two routines
    ds_pred = ds.sel(dataset=DATASET_NAME_PREDICTION)
    ds_test = ds.sel(dataset=DATASET_NAME_FV3_TARGET)
    ds_hires = ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)

    ds_metrics = create_metrics_dataset(ds_pred, ds_test, ds_hires, names)
    ds_metrics.to_netcdf(os.path.join(output_dir, "metrics.nc"))

    # write out yaml file of timesteps used for testing model
    init_times = ds[names["init_time_dim"]].values
    init_times = [vcm.cast_to_datetime(t) for t in init_times]
    with fsspec.open(os.path.join(output_dir, TIMESTEPS_USED_FILENAME), "w") as f:
        yaml.dump(init_times, f)

    # TODO: move this function to another script which creates all the plots
    timesteps_plot_section = plot_timestep_counts(
        output_dir, TIMESTEPS_USED_FILENAME, DPI_FIGURES
    )

    # TODO This should be another script
    metrics_plot_sections = plot_metrics(ds_metrics, output_dir, DPI_FIGURES, names)

    diag_report_sections = plot_diagnostics(
        ds_pred,
        ds_test,
        ds_hires,
        output_dir=output_dir,
        dpi_figures=DPI_FIGURES,
        names=names,
    )

    return {**timesteps_plot_section, **metrics_plot_sections, **diag_report_sections}


def load_data_and_predict_with_ml(
    test_data_path, model_path, high_res_data_path, model_type, names
):
    # get grid
    # because predict_on_test_data loads data and predicts, the easiest way to grab
    # the grid is by loading the test_data again.
    # This redundancy should go away when predict_on_test_data is split apart
    # This is a good demonstration of how functions which do more than one things
    # are inflexible and hard to adapt
    fs = get_fs(test_data_path)
    test_data_urls = sorted(fs.ls(test_data_path))
    mapper = fs.get_mapper(test_data_urls[0])
    ds = xr.open_zarr(mapper)

    # Drop init time dim that originates from its source in train data
    # vcm.select.get_latlon_grid_coords grid input shouldn't have this dim
    grid = (
        safe.get_variables(ds, names["grid_vars"])
        .isel({names["init_time_dim"]: 0})
        .drop(names["init_time_dim"])
    )

    # TODO this function mixes I/O and computation
    # Should just be 1. load_data, 2. make a prediction
    ds_test, ds_pred = predict_on_test_data(
        test_data_path,
        model_path,
        names["pred_vars_to_keep"],
        names["init_time_dim"],
        names["coord_z_center"],
        model_type,
    )

    ds_test = add_column_heating_moistening(
        ds_test,
        names["suffix_coarse_train_diag"],
        names["var_pressure_thickness"],
        names["var_q_moistening_ml"],
        names["var_q_heating_ml"],
        names["coord_z_center"],
    )

    ds_pred = add_column_heating_moistening(
        ds_pred,
        names["suffix_coarse_train_diag"],
        names["var_pressure_thickness"],
        names["var_q_moistening_ml"],
        names["var_q_heating_ml"],
        names["coord_z_center"],
    )

    # TODO Do all data merginig and loading before computing anything
    logger.info("Loading high-resolution diagnostics")
    init_times = list(set(ds_test[names["init_time_dim"]].values))
    ds_hires = load_high_res_diag_dataset(
        high_res_data_path,
        init_times,
        names["init_time_dim"],
        names["renamed_hires_grid_vars"],
    )

    slmsk: xr.DataArray = ds_test[names["var_land_sea_mask"]].isel(
        {names["init_time_dim"]: 0}
    )

    # TODO ditto: do all merging of data before computing anything
    return (
        merge_comparison_datasets(
            data_vars=names["data_vars"],
            datasets=[ds_pred, ds_test, ds_hires],
            dataset_labels=[
                DATASET_NAME_PREDICTION,
                DATASET_NAME_FV3_TARGET,
                DATASET_NAME_SHIELD_HIRES,
            ],
        )
        .merge(slmsk.to_dataset())
        .merge(grid)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", type=str, help="Model file location. Can be local or remote."
    )
    parser.add_argument(
        "test_data_path",
        type=str,
        help="Path to directory containing test data zarrs." "Can be local or remote.",
    )
    parser.add_argument(
        "high_res_data_path",
        type=str,
        help="Path to C48 coarsened high res diagnostic data.",
    )
    parser.add_argument(
        "variable_names_file", type=str, help="yml with variable name information"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output dir to write results to. Can be local or a GCS path.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        help="Type of model to use. Default is random forest 'rf'. "
        "The only type implemented right now is 'rf'.",
    )
    parser.add_argument(
        "--downsample-time-factor",
        type=int,
        default=1,
        help="Factor by which to downsample test set time steps",
    )

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    args.test_data_path = os.path.join(args.test_data_path, "test")
    with open(args.variable_names_file, "r") as f:
        names = yaml.safe_load(f)

    # remove trailing slash since in GCS some/path// is different than some/path/
    output_dir = args.output_path.rstrip("/")

    # TODO Lot's of input arguments and "and" in the name
    # (potential target for refactor) At least it is clear
    # that this can be improved now.
    # Ideally, ML prediction and loading should be refactored to a separate routine
    ds = load_data_and_predict_with_ml(
        args.test_data_path,
        args.model_path,
        args.high_res_data_path,
        args.model_type,
        names,
    )

    # force loading now to avoid I/O issues down the line
    # This could lead to OOM errors (but those sound like an issue anyway)
    ds = ds.load()

    if _is_remote(args.output_path):
        with tempfile.TemporaryDirectory() as local_dir:
            # TODO another "and" indicates this needs to be refactored.
            report_sections = compute_metrics_and_plot(ds, local_dir, names)
            _write_report(local_dir, report_sections, vars(args), REPORT_TITLE)
            copy(local_dir, output_dir)
    else:
        report_sections = compute_metrics_and_plot(ds, args.output_path, names)
        _write_report(args.output_path, report_sections, vars(args), REPORT_TITLE)

import argparse
import os
import shutil
import xarray as xr
import yaml

from vcm.cloud.fsspec import get_fs, get_protocol
from vcm.cloud.gsutil import copy

from ..create_report import create_report
from ..data import merge_comparison_datasets
from .data import (
    predict_on_test_data,
    load_high_res_diag_dataset,
    add_column_heating_moistening,
)
from .diagnostics import plot_diagnostics
from .create_metrics import create_metrics_dataset
from .plot_metrics import plot_metrics

DATASET_NAME_PREDICTION = "prediction"
DATASET_NAME_FV3_TARGET = "C48_target"
DATASET_NAME_SHIELD_HIRES = "coarsened_high_res"

DPI_FIGURES = {
    "LTS": 100,
    "dQ2_pressure_profiles": 100,
    "R2_pressure_profiles": 100,
    "diurnal_cycle": 90,
    "map_plot_3col": 120,
    "map_plot_single": 100,
}


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
        "output_path",
        type=str,
        help="Output dir to write results to. Can be local or a GCS path.",
    )
    parser.add_argument(
        "variable_names_file", type=str, help="yml with variable name information"
    )
    parser.add_argument(
        "--num_test_zarrs",
        type=int,
        default=4,
        help="Number of zarrs to concat together for use as test set.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        help="Type of model to use. Default is random forest 'rf'. "
        "The only type implemented right now is 'rf'.",
    )
    parser.add_argument(
        "--delete-local-results-after-upload",
        type=bool,
        default=False,
        help="If uploading to a remote results dir, delete the local copies"
        " after upload.",
    )
    parser.add_argument(
        "--downsample-time-factor",
        type=int,
        default=1,
        help="Factor by which to downsample test set time steps",
    )
    args = parser.parse_args()
    args.test_data_path = os.path.join(args.test_data_path, "test")
    with open(args.variable_names_file, "r") as f:
        names = yaml.safe_load(f)

    # if output path is remote GCS location, save results to local output dir first
    proto = get_protocol(args.output_path)
    if proto == "" or proto == "file":
        output_dir = args.output_path
    elif proto == "gs":
        remote_data_path, output_dir = os.path.split(args.output_path.strip("/"))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    ds_test, ds_pred = predict_on_test_data(
        args.test_data_path,
        args.model_path,
        args.num_test_zarrs,
        names["pred_vars_to_keep"],
        names["init_time_dim"],
        names["coord_z_center"],
        args.model_type,
        args.downsample_time_factor,
    )

    fs_input = get_fs(args.test_data_path)
    fs_output = get_fs(args.output_path)

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

    init_times = list(set(ds_test[names["init_time_dim"]].values))
    ds_hires = load_high_res_diag_dataset(
        args.high_res_data_path,
        init_times,
        init_time_dim=names["init_time_dim"],
        renamed_hires_grid_vars=names["renamed_hires_grid_vars"],
    )

    grid_path = os.path.join(os.path.dirname(args.test_data_path), "grid_spec.zarr")

    grid = xr.open_zarr(fs_input.get_mapper(grid_path))
    slmsk = (
        ds_test[names["var_land_sea_mask"]]
        .isel({names["init_time_dim"]: 0})
        .drop(names["init_time_dim"])
    )

    ds = merge_comparison_datasets(
        data_vars=names["data_vars"],
        datasets=[ds_pred, ds_test, ds_hires],
        dataset_labels=[
            DATASET_NAME_PREDICTION,
            DATASET_NAME_FV3_TARGET,
            DATASET_NAME_SHIELD_HIRES,
        ],
        grid=grid,
        additional_dataset=slmsk,
    )
    # separate datasets will now have common grid/sfc_type variables and
    # an identifying dataset coordinate
    ds_pred = ds.sel(dataset=DATASET_NAME_PREDICTION)
    ds_test = ds.sel(dataset=DATASET_NAME_FV3_TARGET)
    ds_hires = ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)

    ds_metrics = create_metrics_dataset(ds_pred, ds_test, ds_hires, names)
    ds_metrics.to_netcdf(os.path.join(output_dir, "metrics.nc"))
    metrics_plot_sections = plot_metrics(ds_metrics, output_dir, DPI_FIGURES, names)

    diag_report_sections = plot_diagnostics(
        ds_pred, ds_test, ds_hires, output_dir, DPI_FIGURES, names
    )

    combined_report_sections = {**metrics_plot_sections, **diag_report_sections}
    create_report(combined_report_sections, "ml_offline_diagnostics", output_dir)

    fs_output = get_fs(args.output_path)
    if proto == "gs":
        copy(output_dir, args.output_path)
        if args.delete_local_results_after_upload is True:
            shutil.rmtree(output_dir)

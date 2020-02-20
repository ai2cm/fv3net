import argparse
from datetime import datetime
import fsspec
import os
import xarray as xr

from vcm.cubedsphere.constants import INIT_TIME_DIM
from fv3net.diagnostics.sklearn_model_performance.data_funcs_sklearn import (
    predict_on_test_data,
    load_high_res_diag_dataset,
)
from fv3net.diagnostics.sklearn_model_performance.plotting_sklearn import make_all_plots
from fv3net.diagnostics.create_report import create_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to directory containing test data zarrs. Can be local or remote.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model file location. Can be local or remote.",
    )
    parser.add_argument(
        "--high-res-data-path",
        type=str,
        required=True,
        help="Path to C48 coarsened high res diagnostic data.",
    )
    parser.add_argument(
        "--num-test-zarrs",
        type=int,
        default=5,
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
        "--output-dir-suffix",
        type=str,
        default="sklearn_regression_predictions",
        help="Directory suffix to write files to. Prefixed with today's timestamp.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    output_dir = f"{timestamp}_{args.output_dir_suffix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ds_test, ds_pred = predict_on_test_data(
        args.test_data_path, args.model_path, args.num_test_zarrs, args.model_type
    )
    init_times = list(set(ds_test[INIT_TIME_DIM].values))
    ds_hires = load_high_res_diag_dataset(args.high_res_data_path, init_times)

    grid_path = os.path.join(os.path.dirname(args.test_data_path), "grid_spec.zarr")
    fs = fsspec.filesystem("gs")
    grid = xr.open_zarr(fs.get_mapper(grid_path))
    report_sections = make_all_plots(ds_pred, ds_test, ds_hires, grid, output_dir)

    create_report(report_sections, "ml_model_predict_diagnostics", output_dir)

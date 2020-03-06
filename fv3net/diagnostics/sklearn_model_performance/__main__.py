import argparse
import fsspec
import os
import shutil
import xarray as xr
from vcm.cubedsphere.constants import INIT_TIME_DIM
from fv3net.diagnostics.sklearn_model_performance.data_funcs_sklearn import (
    predict_on_test_data,
    load_high_res_diag_dataset,
    add_integrated_Q_vars,
)
from fv3net.diagnostics.sklearn_model_performance.plotting_sklearn import make_all_plots
from fv3net.diagnostics.create_report import create_report

TEMP_OUTPUT_DIR = "temp_sklearn_prediction_report_output"

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
        "num_test_zarrs",
        type=int,
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
    args = parser.parse_args()

    ds_test, ds_pred = predict_on_test_data(
        args.test_data_path, args.model_path, args.num_test_zarrs, args.model_type
    )
    add_integrated_Q_vars(ds_test)
    add_integrated_Q_vars(ds_pred)
    init_times = list(set(ds_test[INIT_TIME_DIM].values))
    ds_hires = load_high_res_diag_dataset(args.high_res_data_path, init_times)

    grid_path = os.path.join(os.path.dirname(args.test_data_path), "grid_spec.zarr")
    fs_input, _, _ = fsspec.get_fs_token_paths(args.test_data_path)
    grid = xr.open_zarr(fs_input.get_mapper(grid_path))

    fs_output, _, _ = fsspec.get_fs_token_paths(args.output_path)
    fs_output.makedirs(args.output_path, exist_ok=True)

    report_sections = make_all_plots(ds_pred, ds_test, ds_hires, grid, args.output_path)
    create_report(report_sections, "ml_model_predict_diagnostics", args.output_path)
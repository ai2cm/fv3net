import argparse
import fsspec
from vcm.cloud import gsutil
import os
import shutil
import xarray as xr

from vcm.cubedsphere.constants import INIT_TIME_DIM
from fv3net.diagnostics.sklearn_model_performance.data_funcs_sklearn import (
    predict_on_test_data,
    load_high_res_diag_dataset,
    add_integrated_Q_vars,
)
from vcm.fv3_restarts import _split_url
from fv3net.diagnostics.sklearn_model_performance.plotting_sklearn import make_all_plots
from fv3net.diagnostics.create_report import create_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", type=str, help="Model file location. Can be local or remote."
    )
    parser.add_argument(
        "test_data_path",
        type=str,
        help="Path to directory containing test data zarrs. Can be local or remote.",
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
        help="If uploading to a remote results dir, delete the local copies after upload."
    )
    args = parser.parse_args()

    # if output path is remote GCS location, save results to local output dir first
    proto, path = _split_url(args.output_path)
    if proto == "" or proto == "file":
        if proto == "file":
            path = "/" + path
        output_dir = path
    elif proto == "gs":
        remote_data_path, output_dir = os.path.split(path.strip("/"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds_test, ds_pred = predict_on_test_data(
        args.test_data_path, args.model_path, args.num_test_zarrs, args.model_type
    )
    add_integrated_Q_vars(ds_test)
    add_integrated_Q_vars(ds_pred)
    init_times = list(set(ds_test[INIT_TIME_DIM].values))
    ds_hires = load_high_res_diag_dataset(args.high_res_data_path, init_times)

    grid_path = os.path.join(os.path.dirname(args.test_data_path), "grid_spec.zarr")
    fs = fsspec.filesystem("gs")
    grid = xr.open_zarr(fs.get_mapper(grid_path))
    report_sections = make_all_plots(ds_pred, ds_test, ds_hires, grid, output_dir)
    create_report(report_sections, "ml_model_predict_diagnostics", output_dir)

    if proto == "gs":
        gsutil.copy(output_dir, f"gs://{remote_data_path}")
        if args.delete_local_results_after_upload is True:
            shutil.rmtree(output_dir)
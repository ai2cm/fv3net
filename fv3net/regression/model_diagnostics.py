import argparse
from datetime import datetime
import fsspec
from jinja2 import Template
import os
import xarray as xr

from .plotting import make_all_plots
from vcm.convenience import round_time
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    TILE_COORDS,
)
from vcm.calc.thermo import LATENT_HEAT_VAPORIZATION
from vcm.cloud import gsutil
from fv3net.regression.sklearn import MODEL_FILENAME
from fv3net import COARSENED_DIAGS_ZARR_NAME

kg_m2s_to_mm_day = (1e3 * 86400) / 997.0

SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


report_html = Template(
    """
    {% for header, images in sections.items() %}
        <h2>{{header}}</h2>
            {% for image in images %}
                <img src="{{image}}" />
            {% endfor %}
    {% endfor %}
"""
)


def _predict_on_test_data(test_data_path, model_path, num_test_zarrs, model_type="rf"):
    if model_type == "rf":
        from .sklearn.test import load_test_dataset, load_model, predict_dataset

        ds_test = load_test_dataset(test_data_path, num_test_zarrs)
        sk_wrapped_model = load_model(model_path)
        ds_pred = predict_dataset(sk_wrapped_model, ds_test)
        return ds_test.unstack(), ds_pred
    else:
        raise ValueError(
            "Cannot predict using model type {model_type},"
            "only 'rf' is currently implemented."
        )


def _load_high_res_dataset(coarsened_hires_diags_path, init_times):
    full_zarr_path = os.path.join(coarsened_hires_diags_path, COARSENED_DIAGS_ZARR_NAME)
    fs = fsspec.filesystem("gs")
    ds_hires = xr.open_zarr(
        fs.get_mapper(full_zarr_path), consolidated=True
    ).rename({"time": INIT_TIME_DIM})
    ds_hires = ds_hires.assign_coords(
        {
            INIT_TIME_DIM: [round_time(t) for t in ds_hires[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )
    ds_hires = ds_hires.sel({INIT_TIME_DIM: list(set(init_times))})
    if set(ds_hires[INIT_TIME_DIM].values) != set(init_times):
        raise ValueError(
            f"Timesteps {set(init_times)-set(ds_hires[INIT_TIME_DIM].values)}"
            f"are not matched in high res dataset."
        )
    ds_hires["P-E"] = SEC_PER_DAY * (
        ds_hires["PRATEsfc_coarse"]
        - ds_hires["LHTFLsfc_coarse"] / LATENT_HEAT_VAPORIZATION
    )
    return ds_hires


def _upload_report_dir(output_dir, output_path):
#     proto, path = _split_url(output_path)
    fs = fsspec.filesystem(proto)
    fs.copy(output_dir, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Directory path containing trained model. Can be local or remote.",
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
        help="Optional file output location; may be remove or local.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_FILENAME,
        help=f"Name of the trained model object. Defaults to {MODEL_FILENAME}",
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
        default="sklearn_regression",
        help="Directory suffix to write files to. Prefixed with today's timestamp.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    output_dir = f"{timestamp}_{args.output_dir_suffix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ds_test, ds_pred = _predict_on_test_data(
        args.test_data_path, os.path.join(args.model_path, args.model_name), args.num_test_zarrs, args.model_type
    )
    init_times = list(set(ds_test[INIT_TIME_DIM].values))
    ds_hires = _load_high_res_dataset(args.high_res_data_path, init_times)

    grid_path = os.path.join(os.path.dirname(args.test_data_path), "grid_spec.zarr")
    fs = fsspec.filesystem("gs")
    grid = xr.open_zarr(fs.get_mapper(grid_path))
    report_sections = make_all_plots(ds_pred, ds_test, ds_hires, grid, output_dir)

    with open(f"{output_dir}/model_diagnostics.html", "w") as f:
        html = report_html.render(sections=report_sections)
        f.write(html)
        
    if args.output_path:
        gsutil.copy(output_dir, args.output_path)

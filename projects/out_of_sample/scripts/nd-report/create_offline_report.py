from create_online_report import _STATE_SUFFIX
import fsspec
import fv3fit
import hashlib
from lark import logger
from novelty_report_generation_helper import (
    generate_and_save_report,
    OOSModel,
    _get_parser
)
import os
import uuid
import vcm
import xarray as xr
import yaml



def get_diags_offline_suffix(model_name: str) -> str:
    return f"diags_novelty_offline_{model_name}.zarr"


def get_offline_diags(oos_model: OOSModel, ds_path: str) -> xr.Dataset:
    """
    Returns a dataset containing the is_novelty and novelty_score fields that reflect
    the offline behavior of a novelty detector on some other temporal dataset.
    """
    diags_url = os.path.join(
        oos_model.nd_path,
        f"diags_novelty_offline/{hashlib.md5(ds_path.encode()).hexdigest()}",
    )
    fs = vcm.cloud.get_fs(diags_url)
    if fs.exists(diags_url):
        print(
            f"Reading offline novelty data from "
            + f"previous computation, at {diags_url}."
        )
        diags = xr.open_zarr(fsspec.get_mapper(diags_url))
    else:
        state_url = os.path.join(ds_path, _STATE_SUFFIX)
        print(f"Computing offline novelty data from states at {state_url}.")
        ds = xr.open_zarr(fsspec.get_mapper(state_url))
        _, diags = oos_model.nd.predict_novelties(ds)
        mapper = fsspec.get_mapper(diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {diags_url}.")
    return diags


def create_offline_report(args):
    with open(args.config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    run_name = config["run_dataset"]["name"]
    run_url = config["run_dataset"]["url"]

    models = [
        OOSModel(
            model["name"],
            fv3fit.load(model["model_url"]),
            model["model_url"],
            model.get("cutoff", 0)
        ) for model in config["models"]
    ]
    model_diags = {model.name: get_offline_diags(model, run_url) for model in models}

    report_url = config["report_url"]
    if config["append_random_id_to_url"]:
        report_url = os.path.join(report_url, f"report-{uuid.uuid4().hex}")

    metadata = {
        "models": [
            {"name": model.name, "model_url": model.nd_path} for model in models
        ],
        "run_name": run_name,
        "run_url": run_url,
        "report_url": report_url,
    }

    generate_and_save_report(
        models,
        model_diags,
        config.get("has_cutoff_plots", False),
        metadata,
        report_url,
        f"Offline metrics for out-of-sample analysis on {run_name}"
    )


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    create_offline_report(args)

import dataclasses
import fsspec
import fv3fit
from novelty_report_generation_helper import (
    generate_and_save_report,
    OOSModel,
    _get_parser
)
import os
import uuid
import vcm
from vcm.catalog import catalog
import xarray as xr
import yaml

_DIAGS_SUFFIX = "diags.zarr"
_NOVELTY_DIAGS_SUFFIX = "diags_novelty.zarr"
_STATE_SUFFIX = "state_after_timestep.zarr"

@dataclasses.dataclass
class OnlineOOSModel(OOSModel):
    run_path: str


def get_online_diags(oos_model: OnlineOOSModel) -> xr.Dataset:
    """
    Returns a dataset containing the is_novelty and novelty_score fields that reflect
    how a given novelty detector behaved on its online run.
    """
    diags_url = os.path.join(oos_model.run_path, _DIAGS_SUFFIX)
    diags = xr.open_zarr(fsspec.get_mapper(diags_url))
    fsspec.get_mapper
    novelty_diags_url = os.path.join(oos_model.run_path, _NOVELTY_DIAGS_SUFFIX)
    fs = vcm.cloud.get_fs(novelty_diags_url)
    if "is_novelty" in diags.data_vars:
        print(f"Reading online novelty data from model diagnostics, at {diags_url}.")
        return diags
    elif fs.exists(novelty_diags_url):
        print(
            f"Reading online novelty data from "
            + f"previous computation, at {novelty_diags_url}."
        )
        diags = xr.open_zarr(fsspec.get_mapper(novelty_diags_url))
        return diags
    else:
        state_url = os.path.join(oos_model.run_path, _STATE_SUFFIX)
        print(f"Computing online novelty data from states at {state_url}.")
        ds = xr.open_zarr(fsspec.get_mapper(state_url))
        _, diags = oos_model.nd.predict_novelties(ds, cutoff=oos_model.cutoff)
        mapper = fsspec.get_mapper(novelty_diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {novelty_diags_url}.")
        return diags


def create_online_report(args):
    with open(args.config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    models = [
        OnlineOOSModel(
            model["name"],
            fv3fit.load(model["model_url"]),
            model["model_url"],
            model.get("cutoff", 0),
            model["run_url"]
        ) for model in config["models"]
    ]
    model_diags = {model.name: get_online_diags(model) for model in models}

    report_url = config["report_url"]
    if config["append_random_id_to_url"]:
        report_url = os.path.join(report_url, f"report-{uuid.uuid4().hex}")

    metadata = {
        "models": [
            {"name": model.name, "model_url": model.nd_path, "run_url": model.run_path}
            for model in models
        ],
        "report_url": report_url,
    }

    generate_and_save_report(
        models,
        model_diags,
        config.get("has_cutoff_plots", False),
        metadata,
        report_url,
        "Online metrics for out-of-sample analysis"
    )


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    create_online_report(args)

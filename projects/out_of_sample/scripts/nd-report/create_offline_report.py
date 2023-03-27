from novelty_report_generation_helper import (
    generate_and_save_report,
    OOSModel,
    _get_parser,
)
import datetime
import fsspec
import fv3fit
import hashlib
import intake

import os
import uuid
from typing import Optional, Iterable
import vcm
import xarray as xr
import yaml

_STATE_SUFFIX = "state_after_timestep.zarr"


def get_diags_offline_suffix(model_name: str) -> str:
    return f"diags_novelty_offline_{model_name}.zarr"


def get_offline_diags(
    oos_model: OOSModel,
    ds_path: str,
    n_weeks: Optional[int] = None,
    time_sample_freq: Optional[str] = None,
    variables: Optional[Iterable[str]] = None,
) -> xr.Dataset:
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
        # ds = xr.open_zarr(fsspec.get_mapper(state_url), consolidated=True).to_dask()
        ds = intake.open_zarr(state_url, consolidated=True).to_dask()
        if args.variables is not None:
            print(f"Loading variables {variables}")
            ds = ds[variables]

        if n_weeks is not None:
            tstop = ds.time.values[0] + datetime.timedelta(weeks=n_weeks)
            print(f"Computing up to {tstop}")
            ds = ds.sel(time=slice(None, tstop))
        if time_sample_freq is not None:
            print(f"Resampling time to nearest {time_sample_freq}")
            ds = ds.resample(time=time_sample_freq).nearest()
        ds = ds.load()
        # _, diags = oos_model.nd.predict_novelties(ds)
        diags = predict_loop(oos_model, ds)
        mapper = fsspec.get_mapper(diags_url)
        diags.to_zarr(mapper, mode="w", consolidated=True)
        print(f"Saved online novelty data to {diags_url}.")
    return diags


def predict_loop(oos_model, ds):
    # try avoid kernel crash
    timestep_predictions = []
    for t in ds.time.values:
        timestep = ds.sel(time=t)
        print(f"{datetime.datetime.now()}: predicting timestep {timestep.time.item()}")
        timestep_predictions.append(oos_model.nd.predict_novelties(timestep)[1])
    return xr.concat(timestep_predictions, dim="time")


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
            model.get("cutoff", 0),
        )
        for model in config["models"]
    ]
    model_diags = {
        model.name: get_offline_diags(
            model,
            run_url,
            n_weeks=args.n_weeks,
            time_sample_freq=args.time_sample_freq,
            variables=args.variables,
        )
        for model in models
    }

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
        f"Offline metrics for out-of-sample analysis on {run_name}",
    )


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    create_offline_report(args)

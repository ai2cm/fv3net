from typing import Iterable
import numpy as np
import xarray as xr
import os
import vcm.safe
import argparse
import fv3gfs.util

# data directory and ML input names are currently hard-coded as constants

COARSE_OUTPUT_URL = (
    "/Volumes/OWC Envoy Pro EX/gs/vcm-ml-experiments/"
    "2021-01-22-nudge-to-fine-3hr-averages"
)
# COARSE_OUTPUT_URL = "gs://vcm-ml-experiments/2021-01-22-nudge-to-fine-3hr-averages"

INPUT_NAMES = ["surface_geopotential", "cos_zenith_angle", "land_sea_mask"]
ALL_NAMES = set(INPUT_NAMES).union(
    [
        "air_temperature",
        "specific_humidity",
        "lat",
        "lon",
        "air_temperature_tendency_due_to_nudging",
        "specific_humidity_tendency_due_to_nudging",
        "air_temperature_tendency_due_to_model",
        "specific_humidity_tendency_due_to_model",
    ]
)
STACK_DIMS = (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.TILE_DIM)
SAMPLE_DIM_NAME = "sample"


def open_zarr(url: str) -> xr.Dataset:
    fs = vcm.get_fs(url)
    map_ = fs.get_mapper(url)
    return xr.open_zarr(map_, consolidated=True)


def get_random_indices(n: int, i_max: int):
    batch_indices = np.arange(i_max)
    np.random.shuffle(batch_indices)
    return batch_indices[:n]


def open_nudge() -> Iterable[xr.Dataset]:
    url = COARSE_OUTPUT_URL
    assert not url[-1] == "/"
    nudge_url = f"{url}/nudging_tendencies.zarr"
    state_url = f"{url}/state_after_timestep.zarr"
    state = open_zarr(state_url)
    nudge = open_zarr(nudge_url)
    yield from _open_nudge(state, nudge)


def _open_nudge(state, nudge) -> Iterable[xr.Dataset]:
    # hard-coded prognostic names are air_temperature and specific_humidity
    # this function is hard-coded for this dataset source
    # assumptions like the nudging tendencies being available
    # every 3h while state is every 1.5h are hard-coded
    timestep_seconds = 3 * 60 * 60
    window_seconds = 7 * 24 * 60 * 60
    between_window_seconds = (2 * 24 + 9) * 60 * 60
    nt_window = int(window_seconds / timestep_seconds)
    nt_between_window = int(between_window_seconds / timestep_seconds)
    subsample_fraction = 0.125  # fraction of data to keep, randomly selected

    # state is available on twice the timestep as nudging. On the second timestep,
    # it corresponds to the state after the first data point of nudging has been applied
    state = state.isel(time=range(1, len(state["time"]), 2)).rename_vars(
        {"longitude": "lon", "latitude": "lat"}
    )
    state = vcm.DerivedMapping(state).dataset(
        list(state.data_vars.keys()) + ["cos_zenith_angle"]
    )
    assert nudge["time"][0] < state["time"][0]
    nudge = nudge.isel(time=slice(1, None))

    time = state["time"]
    timestep_seconds = (time[1].values.item() - time[0].values.item()).total_seconds()

    T_tendency = (
        state["air_temperature"].isel(time=slice(1, None)).values
        - state["air_temperature"].isel(time=slice(0, -1)).values
    ) / timestep_seconds - nudge["air_temperature_tendency_due_to_nudging"].values
    q_tendency = (
        state["specific_humidity"].isel(time=slice(1, None)).values
        - state["specific_humidity"].isel(time=slice(0, -1)).values
    ) / timestep_seconds - nudge["specific_humidity_tendency_due_to_nudging"].values

    state = state.isel(time=slice(0, -1))
    state["air_temperature_tendency_due_to_model"] = (
        state["air_temperature"].dims,
        T_tendency,
        {"units": state["air_temperature"].attrs["units"] + " / s"},
    )
    assert np.sum(np.isnan(state["air_temperature_tendency_due_to_model"].values)) == 0
    state["specific_humidity_tendency_due_to_model"] = (
        state["specific_humidity"].dims,
        q_tendency,
        {"units": state["specific_humidity"].attrs["units"] + " / s"},
    )
    nudge = nudge.assign_coords({"time": state["time"]})
    ds = xr.merge([nudge, state], join="inner")
    drop_vars = set(name for name in ds.data_vars if name not in ALL_NAMES)
    ds = ds.drop_vars(drop_vars)
    ds = vcm.safe.stack_once(ds, SAMPLE_DIM_NAME, STACK_DIMS).reset_index(
        SAMPLE_DIM_NAME
    )
    # put data in ML order (sample, time, feature)
    ds = ds.transpose(SAMPLE_DIM_NAME, "time", "z")
    n_total_samples = len(ds[SAMPLE_DIM_NAME])
    n_samples = int(subsample_fraction * n_total_samples)
    nt = len(ds["time"])

    for i_start in range(0, nt - nt_window, nt_between_window):
        sample_idx = get_random_indices(n_samples, n_total_samples)
        ds_window = ds.isel(time=slice(i_start, i_start + nt_window)).isel(
            **{SAMPLE_DIM_NAME: sample_idx}
        )
        yield ds_window


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_data_path", type=str, help="directory to output array data"
    )
    return parser


if __name__ == "__main__":
    np.random.seed(0)
    parser = get_parser()
    args = parser.parse_args()
    fs = vcm.get_fs(args.output_data_path)
    fs.makedirs(args.output_data_path, exist_ok=True)
    for i, ds in enumerate(open_nudge()):
        filename = os.path.join(args.output_data_path, f"window_{i:05d}.nc")
        print(f"saving {filename}...")
        with fs.open(filename, "wb") as f:
            ds.to_netcdf(f)

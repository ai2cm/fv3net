import sys
import intake
import xarray as xr
import os
from dask.diagnostics import ProgressBar
import fsspec
import yaml

from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm import convert_timestamps


COARSE_NUDGED_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-09-14/cloud-ml-training-data-nudge-to-fine-v5/fv3gfs_run"  # noqa: E501
COARSE_STATE_ZARR = "state_after_timestep.zarr"
GRID = CATALOG["grid/c48"].to_dask()
MASK = CATALOG["landseamask/c48"].to_dask()
FINE_RESTARTS_KEYS = [
    "40day_c48_restarts_as_zarr_may2020",
    "40day_c48_gfsphysics_15min_may2020",
]
FINE_TO_COARSE_RENAME = {
    "T": "air_temperature",
    "sphum": "specific_humidity",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "phis": "surface_geopotential",
    "liq_wat": "cloud_water_mixing_ratio",
    "ice_wat": "cloud_ice_mixing_ratio",
    "rainwat": "rain_mixing_ratio",
    "snowwat": "snow_mixing_ratio",
    "graupel": "graupel_mixing_ratio",
    "cld_amt": "cloud_amount",
}
RENAME_DIMS = {"pfull": "z"}
COORD_VARS = ["x", "y", "z", "tile"]
OUTPUT_CHUNKS = {"time": 1, "tile": 6}


def get_fine_ds(keys, fine_rename):
    datasets = []
    for key in keys:
        dataset = CATALOG[key].to_dask()
        if isinstance(dataset.time[0].item(), str):
            dataset = dataset.assign_coords({"time": convert_timestamps(dataset.time)})
        dataset = standardize_fv3_diagnostics(dataset)
        if "pfull" in dataset.dims:
            dataset = dataset.rename(RENAME_DIMS)
        datasets.append(dataset)
    ds = xr.merge(datasets)
    ds_out = xr.Dataset()
    for restart_name, python_name in fine_rename.items():
        ds_out[python_name] = ds[restart_name]
    return ds_out.drop_vars(COORD_VARS)


def get_coarse_ds(path, zarr, coarse_rename):
    full_path = os.path.join(path, zarr)
    ds = intake.open_zarr(full_path, consolidated=True).to_dask()
    ds_out = xr.Dataset()
    for old_name, new_name in coarse_rename.items():
        ds_out[new_name] = ds[old_name]
    return ds_out


def subset_times(coarse, fine):
    common_times = xr.DataArray(
        data=sorted(list(set(coarse.time.values).intersection(set(fine.time.values)))),
        dims=["time"],
    )
    return coarse.sel(time=common_times), fine.sel(time=common_times)


def rechunk(ds, chunks=OUTPUT_CHUNKS):
    for var in ds:
        ds[var].encoding = {}
    return ds.unify_chunks().chunk(chunks)


def get_config(path):
    if path is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        return None


def main(config):
    """
    Prescribers and fv3fit ML models need training data in a particular form.
    e.g., let's make a dataset of the following targets:

    - air temperature
    - specific humidity
    - layer height thickness
    - cloud water
    - cloud ice
    - graupel water
    - rain water
    - snow water
    - cloud fraction
    """

    fine = get_fine_ds(
        keys=config.get("fine_keys", FINE_RESTARTS_KEYS),
        fine_rename=config.get("fine_rename", FINE_TO_COARSE_RENAME),
    )
    coarse = get_coarse_ds(
        path=config.get("coarse_nudged_path", COARSE_NUDGED_PATH),
        zarr=config.get("coarse_state_zarr", COARSE_STATE_ZARR),
        coarse_rename=config.get("coarse_rename", FINE_TO_COARSE_RENAME),
    )
    coarse, fine = subset_times(coarse, fine)
    assert (
        len(set(coarse.data_vars).intersection(fine.data_vars)) == 0
    ), "non-overlapping names"
    merged = xr.merge([fine, coarse, GRID, MASK])
    merged = rechunk(merged)
    try:
        output_path = config["output_path"]
    except KeyError:
        raise KeyError('Config must contain "output_path".')
    with ProgressBar():
        print(f'Number of timesteps: {merged.sizes["time"]}.')
        print(f"Data variables: {[var for var in merged.data_vars]}")
        merged.to_zarr(fsspec.get_mapper(output_path), consolidated=True)


if __name__ == "__main__":
    if len(sys.argv[1]) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None
    config = get_config(config_path)
    main(config)

import intake
import xarray as xr
import os
from dask.diagnostics import ProgressBar
import fsspec

from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm import convert_timestamps


COARSE_NUDGED_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-09-14/cloud-ml-training-data-nudge-to-fine-v5/fv3gfs_run"  # noqa: E501
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
OUTPUT_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-09-14/fine-coarse-3d-fields.zarr"


def get_fine_ds():
    datasets = []
    for key in FINE_RESTARTS_KEYS:
        dataset = CATALOG[key].to_dask()
        if isinstance(dataset.time[0].item(), str):
            dataset = dataset.assign_coords({"time": convert_timestamps(dataset.time)})
        dataset = standardize_fv3_diagnostics(dataset)
        if "pfull" in dataset.dims:
            dataset = dataset.rename(RENAME_DIMS)
        datasets.append(dataset)
    ds = xr.merge(datasets)
    ds_3d = xr.Dataset()
    for restart_name, python_name in FINE_TO_COARSE_RENAME.items():
        fine_name = python_name + "_fine"
        ds_3d[fine_name] = ds[restart_name]
    return ds_3d.drop_vars(COORD_VARS)


def get_coarse_ds():
    full_path = os.path.join(COARSE_NUDGED_PATH, "state_after_timestep.zarr")
    ds = intake.open_zarr(full_path, consolidated=True).to_dask()
    ds_3d = xr.Dataset()
    for var in FINE_TO_COARSE_RENAME.values():
        coarse_name = var + "_coarse"
        ds_3d[coarse_name] = ds[var]
    return ds_3d


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


def main():
    """
    Prescribers and fv3fit ML models need training data in a particular form.
    Let's make a dataset of the following targets:

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

    fine = get_fine_ds()
    coarse = get_coarse_ds()
    coarse, fine = subset_times(coarse, fine)
    merged = xr.merge([fine, coarse, GRID, MASK])
    merged = rechunk(merged)
    with ProgressBar():
        print(f'Number of timesteps: {merged.sizes["time"]}.')
        merged.to_zarr(fsspec.get_mapper(OUTPUT_PATH), consolidated=True)


if __name__ == "__main__":
    main()

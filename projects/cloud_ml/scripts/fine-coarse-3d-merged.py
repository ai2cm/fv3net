import intake
import xarray as xr
import os
from dask.diagnostics import ProgressBar
import fsspec

from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm import convert_timestamps
from vcm.cubedsphere import center_and_rotate_xy_winds
from vcm.safe import get_variables


WIND_ROTATION_MATRIX = CATALOG["wind_rotation/c48"].to_dask()
GRID = CATALOG["grid/c48"].to_dask()
COARSE_NUDGED_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-06-24/cloud-ml-training-data-trial-0/fv3gfs_run"  # noqa: E501
FINE_RESTARTS_KEY = "40day_c48_restarts_as_zarr_may2020"
FINE_TO_COARSE_RENAME = {
    "T": "air_temperature",
    "sphum": "specific_humidity",
    "eastward_wind": "eastward_wind",
    "northward_wind": "northward_wind",
    "DZ": "vertical_thickness_of_atmospheric_layer",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "liq_wat": "cloud_water_mixing_ratio",
    "ice_wat": "cloud_ice_mixing_ratio",
    "rainwat": "rain_mixing_ratio",
    "snowwat": "snow_mixing_ratio",
    "graupel": "graupel_mixing_ratio",
    "cld_amt": "cloud_amount",
}
RENAME_DIMS = {"pfull": "z"}
COORD_VARS = ["x", "y", "z", "tile"]
OUTPUT_CHUNKS = {"tile": 6}
OUTPUT_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-06-24/fine-coarse-3d-fields.zarr"


def rotate_winds(ds):
    eastward_wind, northward_wind = center_and_rotate_xy_winds(
        WIND_ROTATION_MATRIX, ds.u, ds.v
    )
    ds["eastward_wind"] = eastward_wind
    ds["northward_wind"] = northward_wind
    return ds.drop_vars(["u", "v"])


def get_fine_ds():
    ds = CATALOG[FINE_RESTARTS_KEY].to_dask()
    ds = ds.assign_coords({"time": convert_timestamps(ds.time)})
    ds = standardize_fv3_diagnostics(ds).rename(RENAME_DIMS)
    rotate_winds(ds)
    ds_3d = xr.Dataset()
    for restart_name, python_name in FINE_TO_COARSE_RENAME.items():
        ds_3d[python_name] = ds[restart_name]
    return ds_3d.drop_vars(COORD_VARS)


def get_coarse_ds():
    full_path = os.path.join(COARSE_NUDGED_PATH, "state_after_timestep.zarr")
    ds = intake.open_zarr(full_path, consolidated=True).to_dask()
    ds_3d = xr.Dataset()
    for var in FINE_TO_COARSE_RENAME.values():
        ds_3d[var] = ds[var]
    return ds_3d


def merge_coarse_fine(coarse, fine):
    # subset times and variables to intersection
    common_times = xr.DataArray(
        data=sorted(list(set(coarse.time.values).intersection(set(fine.time.values)))),
        dims=["time"],
    )
    coarse, fine = coarse.sel(time=common_times), fine.sel(time=common_times)
    common_vars = set(coarse.data_vars).intersection(fine.data_vars)
    coarse, fine = get_variables(coarse, common_vars), get_variables(fine, common_vars)

    # add 'res' dimension and concatenate
    merged = xr.concat(
        [coarse.expand_dims({"res": ["coarse"]}), fine.expand_dims({"res": ["fine"]})],
        dim="res",
    )
    # add grid variables back
    return xr.merge([merged, GRID])


def main():
    """
    Let's make a merged dataset of fine and coarse 3D (really, time-tile-x-y-z)
    variables, including the water tracer species. Specifically:

    - air temperature
    - specific humidity
    - northward wind
    - eastward wind
    - layer pressure thickness
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
    merged = merge_coarse_fine(coarse, fine)

    merged = merged.unify_chunks().chunk(OUTPUT_CHUNKS)
    with ProgressBar():
        merged.to_zarr(fsspec.get_mapper(OUTPUT_PATH), consolidated=True)


if __name__ == "__main__":
    main()

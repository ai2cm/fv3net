# flake8: noqa
import xarray as xr
from datetime import datetime, timedelta
import vcm
import cftime
import numpy as np
import os

# fine-res h500, PRESsfc, w500, TB, TMP500_300, and PWAT is at gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/
# fine-res PRATEsfc_coarse is at "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"

# coarse-res data is interpolated to a regular 3h grid, since runs did not evenly divide
# by 3h. Data for h500, PRESsfc, w500, TMPlowest, RH850, RH500, TMP500_300, and PWAT is available at e.g.
# gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/atmos_dt_atmos.zarr
# coarse-res PRATEsfc is available at gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/sfc_dt_atmos.zarr/
# similar coarse-res data is available in other subdirectories

VARNAMES = [
    "PRATEsfc",
]


def convert_fine(ds_precip: xr.Dataset) -> xr.Dataset:
    ds_precip = ds_precip.rename(
        {
            "PRATEsfc_coarse": "PRATEsfc",
            "grid_xt_coarse": "grid_xt",
            "grid_yt_coarse": "grid_yt",
        }
    )
    return ds_precip


def convert_coarse(ds_precip: xr.Dataset) -> xr.Dataset:
    return ds_precip


def write(fine: xr.Dataset, coarse: xr.Dataset, name: str):
    date_string = datetime.now().strftime("%Y-%m-%d")
    fine_path = f"gs://vcm-ml-experiments/mcgibbon/{date_string}/fine-{name}.zarr"
    if not vcm.get_fs(fine_path).exists(fine_path):
        print(f"writing fine-res data to {fine_path}")
        fine.to_zarr(fine_path, consolidated=True)
    else:
        print(f"fine-res data already exists at {fine_path}, skipping")
    coarse_path = f"gs://vcm-ml-experiments/mcgibbon/{date_string}/coarse-{name}.zarr"
    if not vcm.get_fs(coarse_path).exists(coarse_path):
        print(f"writing coarse-res data to {coarse_path}")
        coarse.to_zarr(coarse_path, consolidated=True)
    else:
        print(f"coarse-res data already exists at {coarse_path}, skipping")


coarse_base_paths = [
    "gs://vcm-ml-raw-flexible-retention/2023-02-23-CycleGAN-reference-simulations/C48-0K/sfc_8xdaily.zarr",
]
fine_base_paths = [
    "gs://vcm-ml-raw-flexible-retention/2023-02-23-CycleGAN-reference-simulations/C384-0K/{:04d}010100/sfc_8xdaily_coarse.zarr",
]
path_labels = ["baseline"]


def drop_vars(ds: xr.Dataset):
    drop_names = [name for name in ds.data_vars.keys() if name not in VARNAMES]
    return ds.drop_vars(drop_names)


if __name__ == "__main__":
    fine_list = []
    coarse_list = []
    for fine_path, coarse_path, label in zip(
        fine_base_paths, coarse_base_paths, path_labels
    ):
        fine_datasets = []
        for i in range(2017, 2025):
            ds_fine = xr.open_zarr(fine_path.format(i))
            ds_fine = drop_vars(convert_fine(ds_fine))
            # discard first month of spin-up time
            ds_fine = ds_fine.isel(time=slice(8 * 31, None))
            fine_datasets.append(ds_fine)
        ds_fine = xr.concat(fine_datasets, dim="time")
        ds_coarse = xr.open_zarr(coarse_path)
        ds_coarse = drop_vars(convert_coarse(ds_coarse))
        # discard first year + month of spin-up time, starts on leap year
        ds_coarse = ds_coarse.isel(time=slice(8 * 31 + 8 * 366, None))
        fine_list.append(ds_fine)
        coarse_list.append(ds_coarse)
    fine_nt_min = np.inf
    coarse_nt_min = np.inf
    for ds in fine_list:
        fine_nt_min = min(fine_nt_min, len(ds.time))
    for ds in coarse_list:
        coarse_nt_min = min(coarse_nt_min, len(ds.time))
    fine_list = [ds.isel(time=slice(fine_nt_min)) for ds in fine_list]
    coarse_list = [ds.isel(time=slice(coarse_nt_min)) for ds in coarse_list]
    ds_fine = xr.concat(fine_list, dim="perturbation").assign_coords(
        {"perturbation": path_labels}
    )
    ds_coarse = xr.concat(coarse_list, dim="perturbation").assign_coords(
        {"perturbation": path_labels}
    )
    ds_fine = drop_vars(ds_fine)
    ds_coarse = drop_vars(ds_coarse)
    ds_fine = ds_fine.chunk({"time": 365, "tile": 6, "grid_xt": 48, "grid_yt": 48})
    ds_coarse = ds_coarse.chunk({"time": 365, "tile": 6, "grid_xt": 48, "grid_yt": 48})
    print(ds_coarse, ds_fine)
    write(ds_fine, ds_coarse, "combined-march")

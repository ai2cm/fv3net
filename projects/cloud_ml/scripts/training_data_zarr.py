import sys
import xarray as xr
import os
from dask.diagnostics import ProgressBar
import yaml

from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm import convert_timestamps


GRID = CATALOG["grid/c48"].to_dask()
MASK = CATALOG["landseamask/c48"].to_dask()
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
            dataset = dataset.rename({"pfull": "z"})
        datasets.append(dataset)
    ds = xr.merge(datasets)
    ds_out = xr.Dataset()
    for restart_name, python_name in fine_rename.items():
        ds_out[python_name] = ds[restart_name]
    return ds_out.drop_vars(COORD_VARS)


def get_coarse_ds(path, zarr, coarse_rename):
    ds = xr.open_zarr(os.path.join(path, zarr))
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
    fv3fit needs training data in a particular form;
    create suitable zarr of inputs and outputs
    """

    fine = get_fine_ds(keys=config["fine_keys"], fine_rename=config["fine_rename"],)
    coarse = get_coarse_ds(
        path=config["coarse_nudged_path"],
        zarr=config["coarse_state_zarr"],
        coarse_rename=config["coarse_rename"],
    )
    coarse, fine = subset_times(coarse, fine)
    assert (
        len(set(coarse.data_vars).intersection(fine.data_vars)) == 0
    ), "non-overlapping names"
    merged = xr.merge([fine, coarse, GRID, MASK])
    merged = rechunk(merged)
    output_path = config["output_path"]
    with ProgressBar():
        print(f'Number of timesteps: {merged.sizes["time"]}.')
        print(f"Data variables: {[var for var in merged.data_vars]}")
        merged.to_zarr(output_path, consolidated=True)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    main(config)

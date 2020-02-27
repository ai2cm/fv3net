from vcm.convenience import open_delayed
import vcm
from dask.delayed import delayed
import pandas as pd
import xarray as xr
import fsspec
import os
from pathlib import Path

from vcm.fv3_restarts import _parse_time_string

# from distributed import Client

# client = Client("tcp://192.168.100.88:8786")


@delayed
def open_restarts_eager(url):
    return vcm.open_restarts_with_time_coordinates(url).load()


def change_to_forecast_time(ds):
    ds_dt = ds.assign_coords(time=ds.time - ds.time[0])
    return ds_dt.rename({'time': 'forecast_time'})


def _parse_time_from_base(path):
    timestep = str(Path(path).name)
    return _parse_time_string(timestep)




fs = fsspec.filesystem('gs')

url = "gs://vcm-ml-data/orchestration-testing/test-andrep/one_step_run_one_step_yaml_all-physics-off.yml_experiment_label_test-orchestration-group/"


if __name__ == '__main__':
    timesteps = ['gs://' + _ for _ in fs.ls(url)]

    initial_times = []
    delayed_objs = []
    run_folder = []

    for step in timesteps:
        try:
            time = _parse_time_from_base(step)
        except ValueError:
            pass
        else:
            obj = open_restarts_eager(step)
            delayed_objs.append(obj)
            initial_times.append(time)
            run_folder.append(step)


    # load one file shema
    schema = vcm.open_restarts_with_time_coordinates(run_folder[0])

    # apply metadata
    datasets = [open_delayed(obj, schema) for obj in delayed_objs]

    # change time to delta time
    datasets_no_time = [change_to_forecast_time(ds) for ds in datasets]

    merged = xr.concat(
        datasets_no_time,
        dim=pd.Index(run_folder, name='run_folder')
    )

    initial_times = xr.DataArray(initial_times, dims=['run_folder'])

    cleaned = merged.assign_coords(initial_time=initial_times).swap_dims({'run_folder': 'initial_time'})

    # cleaned.to_zarr("big.zarr", mode="w")

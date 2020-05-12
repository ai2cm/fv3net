import apache_beam as beam
import joblib
import xarray as xr
import datetime
import os
from itertools import product
import fsspec
import merge_restarts_and_diags as merge

# from vcm import safe
import dask
import budgets

dask.config.set(scheduler="single-threaded")

import logging


def yield_all(ds, times, tiles):
    for time, tile in product(times, tiles):
        local = ds.isel(time=time, tile=tile)
        for variable in ds:
            yield (ds.time[time].item(), tile), local[variable]


def load_data_array(da):
    return da.load()


def open_remote_zarr(url):
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def ValMap(func, *args, **kwargs):
    """Map over values
    """

    def myfunc(keyval, *args, **kwargs):
        key, val = keyval
        logging.info(f"Calling {func} on {key}")
        return key, func(val, *args, **kwargs)

    return beam.Map(myfunc, *args, **kwargs)


def KeyMap(func, *args, **kwargs):
    """Map over values
    """

    def myfunc(keyval, *args, **kwargs):
        key, val = keyval
        return func(key, *args, **kwargs), val

    return beam.Map(myfunc, *args, **kwargs)


def write_to_disk(keyval, path):
    key, val = keyval
    time, tile = key

    name = val.name
    dirname = f"{path}/{time}/"
    os.makedirs(dirname, exist_ok=True)
    path = f"{path}/{time}/{name}.tile{tile}.nc"
    logging.info(f"saving to {path}")
    val.to_netcdf(path)


def shift_time(keyval, dt):
    (time, tile), val = keyval
    return (time + dt, tile), val.drop("time")


def save(keyval, area):
    (time, tile), val = keyval
    fname = f"{time}/{tile}.nc"
    os.makedirs(f"{time}", exist_ok=True)
    joblib.dump((keyval, area), fname)


def budget(keyval, area):
    (time, tile), val = keyval

    grid = ["grid_xt", "grid_yt", "tile"]
    area = area.sel(tile=tile).drop(grid)

    begin = xr.merge(val["begin"])
    end = xr.merge(val["end"])
    phys = xr.merge(val["phys"]).drop(grid)
    return keyval, budgets.compute_recoarsened_budget_v2(begin, end, phys, area)


def is_complete(kv):
    _, d = kv
    for name in d:
        if len(d[name]) == 0:
            return False
    return True


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    physics_url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/"
    restart_url = (
        "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr"
    )

    times_diag = [1]
    times_restart = [0, 1]
    tiles = [0]

    PHYSICS_VARIABLES = [
        "omega",
        "t_dt_gfdlmp",
        "t_dt_nudge",
        "t_dt_phys",
        "qv_dt_gfdlmp",
        "qv_dt_phys",
        "eddy_flux_omega_sphum",
        "eddy_flux_omega_temp",
    ]

    FROM_DIAG_RENAME = {""}

    from apache_beam.options.pipeline_options import PipelineOptions

    # options = PipelineOptions(["--runner=FlinkRunner"])
    options = None

    with beam.Pipeline(options=options) as p:

        physics_ds = (
            p
            | "Diagnostic Url" >> beam.Create([physics_url])
            | "Open D" >> beam.Map(open_remote_zarr)
            | "StandardizePhysics" >> beam.Map(merge.standardize_diagnostic_metadata)
        )

        area = physics_ds | "GetArea" >> beam.Map(lambda ds: ds.area.load())

        physics_data = (
            physics_ds
            | "AllPhysicsTiles" >> beam.ParDo(yield_all, times_diag, tiles)
            | "SelectV" >> beam.Filter(lambda x: x[1].name in PHYSICS_VARIABLES)
            | "LoadArrayP" >> ValMap(load_data_array)
        )

        restarts_ = (
            p
            | "Restart" >> beam.Create([restart_url])
            | "Open Z Restart" >> beam.Map(open_remote_zarr)
            | "StandardizeRestartMetadata"
            >> beam.Map(merge.standardize_restart_metadata)
            | "AllRestartTiles" >> beam.ParDo(yield_all, times_restart, tiles)
            | "LoadArray" >> ValMap(load_data_array)
        )

        # shift the restart times
        dt = datetime.timedelta(minutes=7, seconds=30)
        begin = restarts_ | "Shift Forward" >> beam.Map(shift_time, -dt)
        end = restarts_ | "Shift Back" >> beam.Map(shift_time, dt)

        (
            {"end": end, "begin": begin, "phys": physics_data,}
            | beam.CoGroupByKey()
            | "FilterComplete" >> beam.Filter(is_complete)
            | "Save" >> beam.Map(save, beam.pvalue.AsSingleton(area))
        )

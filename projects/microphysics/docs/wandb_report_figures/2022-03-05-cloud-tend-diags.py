# flake8: noqa
# coding: utf-8
from fv3net.diagnostics.prognostic_run.emulation import tendencies
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from vcm.fv3.metadata import *
import fsspec
import fv3viz
import matplotlib.pyplot as plt
import numpy as np
import vcm
import vcm.catalog
import xarray


def get_segmented_cmap(n):
    blues = cm.get_cmap("Blues", n)
    reds = cm.get_cmap("Reds", n)
    cb = blues(np.linspace(0, 1, n))
    cr = reds(np.linspace(0.4, 1, n))
    concat = np.concatenate([cr[::-1, :], cb], axis=0)
    return ListedColormap(concat)


def unique_time(ds):
    unique_inds = sorted(
        [list(ds.time.values).index(time) for time in set(ds.time.values)]
    )
    return ds.isel(time=unique_inds)


def open(url):
    mapper = fsspec.get_mapper(url, use_listings_cache=False, skip_instance_cache=True)
    ds = xr.open_zarr(mapper)
    return unique_time(gfdl_to_standard(ds))


def open_grid():
    return vcm.catalog.catalog["grid/c48"].to_dask().load()


def open_sfc(url):
    return open(url + "/atmos_dt_atmos.zarr")


def open_piggy(url):
    return open(url + "/piggy.zarr")


def open_state(url):
    return unique_time(open(url + "/state_after_timestep.zarr").merge(open_grid()))


def open_rundir(url):
    return unique_time(open_piggy(url).merge(open_state(url)).merge(open_grid()))


def get_tends(zonal_avg):
    def run_tend(tend, ds, name, source):
        if source == "diff":
            return tend(ds, name, "emulator") - tend(ds, name, "physics")
        else:
            return tend(ds, name, source)

    return vcm.combine_array_sequence(
        [
            (name, (source, tend.__name__), run_tend(tend, zonal_avg, name, source))
            for name in ["cloud_water", "air_temperature", "specific_humidity"]
            for source in ["emulator", "physics"]
            for tend in [
                tendencies.gscond_tendency,
                tendencies.precpd_tendency,
                tendencies.total_tendency,
            ]
        ],
        labels=["source", "tend"],
    )


url = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-05/limit-tests-all-loss-rnn-7ef273-30d-v1-gscond-full-online"
ds = open_rundir(url)

ds["cloud_negative"] = ds.cloud_water_mixing_ratio < 0
zonal_avg = vcm.zonal_average_approximate(ds.lat, ds.isel(time=slice(0, 8)))
meridional_slice = vcm.interpolate_unstructured(
    ds, vcm.select.meridional_ring(lon=180)
).swap_dims({"sample": "lat"})

time = 3
plt.figure()
newmp = get_segmented_cmap(n=4)
zonal_avg.cloud_water_mixing_ratio.isel(time=time).plot(
    y="z", yincrease=False, cmap=newmp, vmax=5e-5
)
plt.savefig("cloud.png")
plt.figure()
meridional_slice.cloud_water_mixing_ratio.isel(time=time).plot(
    y="z", yincrease=False, cmap=newmp, vmax=5e-5
)
plt.savefig("cloud_lon0.png")

get_tends(zonal_avg.isel(time=time)).cloud_water.plot(
    row="source", col="tend", y="z", yincrease=False, vmax=1e-8
)

plt.savefig("tendencies.png")

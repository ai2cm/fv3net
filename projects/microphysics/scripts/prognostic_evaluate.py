import os
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from itertools import product

from plot_utils import plot_var, plot_meridional
from vcm import DerivedMapping

name = "dense-tendency-limited"
baseline_name = "spunup-baseline-simple-phys-hybrid-edmf-extended"
path = f"gs://vcm-ml-scratch/andrep/2021-10-02-wandb-training/prognostic-runs/all-tends-limited-dense"
baseline_path = f"gs://vcm-ml-experiments/andrep/2021-05-28/{baseline_name}/fv3gfs_run"


prog = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "state_after_timestep.zarr")), consolidated=True)
phys_baseline = xr.open_zarr(fsspec.get_mapper(os.path.join(baseline_path, "state_after_timestep.zarr")), consolidated=True)

prog_w_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "state_output.zarr")))

prog_sfc_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "sfc_dt_atmos.zarr")), consolidated=True)
base_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(baseline_path, "sfc_dt_atmos.zarr")), consolidated=True)


# ## Global average time x height of SPHUM and T


def get_vmin_vmax(var, da):
    
    if "mixing_ratio" in var:
        threshold = 98
    else:
        threshold = 99.5
    vmax = np.percentile(da, threshold)
    if np.any(da.values < 0):
        vmin = -vmax
    else:
        vmin = np.percentile(da, 100 - threshold)
        
    return vmin, vmax


def consistent_time_len(*da_args):
    # assumes same start time and time delta
    times = [len(da.time) for da in da_args]
    min_len = np.min(times)
    return [da.isel(time=slice(0, min_len)) for da in da_args]
    

def plot_time_vert_panels(da1, da2, vmin=None, vmax=None, dpi=80):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)
    fig.set_dpi(dpi)
    
    if vmin is None and vmax is None:
        vmin, vmax = get_vmin_vmax(da2.name, da2)
        vmin = 0
    
    da1, da2 = consistent_time_len(da1, da2)
    vkw = dict(vmin=vmin, vmax=vmax)
    da1.plot.pcolormesh(x="time", y="z", ax=ax[0], yincrease=False, **vkw)
    da2.plot.pcolormesh(x="time", y="z", ax=ax[1], yincrease=False, **vkw)
    (da1 - da2).plot.pcolormesh(x="time", y="z", ax=ax[2], yincrease=False)
    ax[0].set_title("Emulation")
    ax[1].set_title("Baseline")
    ax[2].set_title("Diff")
    
    for sub_ax in ax:
        sub_ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    plt.show()


def get_avg_data(exp_name, ds, filename="after_timestep_mean_by_height.nc", avg_dims=None):
    preproc_path = f"preproc_exp/{exp_name}/{filename}"
    if avg_dims is None:
        avg_dims = ["tile", "x", "y"]
        
    if os.path.exists(preproc_path):
        prog_avg = xr.open_dataset(preproc_path)
    else:
        prog_avg = ds.mean(dim=avg_dims)
        with ProgressBar():
            prog_avg.load()
        os.makedirs(os.path.dirname(preproc_path), exist_ok=True)
        prog_avg.to_netcdf(preproc_path)
        
    return prog_avg


prog_mean_by_height = get_avg_data(name, prog)
base_mean_by_height = get_avg_data(baseline_name, phys_baseline)


plot_time_vert_panels(
    prog_mean_by_height["cloud_water_mixing_ratio"],
    base_mean_by_height["cloud_water_mixing_ratio"]
)


plot_time_vert_panels(
    prog_mean_by_height["specific_humidity"],
    base_mean_by_height["specific_humidity"]
)


plot_time_vert_panels(
    prog_mean_by_height["air_temperature"],
    base_mean_by_height["air_temperature"],
    vmin=220, vmax=290
)


plot_time_vert_panels(
    prog_mean_by_height["eastward_wind"],
    base_mean_by_height["eastward_wind"],
)


plot_time_vert_panels(
    prog_mean_by_height["northward_wind"],
    base_mean_by_height["northward_wind"],
)


# ## Global average comparison after timestep
prog_mean = prog_mean_by_height.mean(dim=["z"]).compute()
base_mean = base_mean_by_height.mean(dim=["z", "z_soil"]).compute()

for var, da in prog_mean.items():
    if var in ["land_sea_mask", "latitude", "longitude"]:
        continue
    if var not in base_mean:
        print(f"Skipping: {var}")
        continue
    
    fig, ax = plt.subplots()
    fig.set_dpi(80)
        
    avg_dims = [dim for dim in da.dims if dim != "time"]
    da.plot(ax=ax, label="Emulator")
    base_mean[var].sel(time=slice(None, da.time.isel(time=-1))).plot(ax=ax, label="Baseline", alpha=0.5)
    plt.legend()
    plt.show()


# ## Meridional transects of condensate and vapor
def target_prediction_2panel_meridional(emu, base, var_names, time_idx=0):
    print(emu.time[time_idx])
    for var in var_names:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 4)
        fig.set_dpi(80)
        
        # get pcolor bounds
        t0_emu = emu[var].isel(time=time_idx)
        vmax = np.percentile(t0_emu, 99.5)
        vmin = -vmax
        
        plot_meridional(t0_emu, var, ax=ax[0], vmin=vmin, vmax=vmax)
        plot_meridional(base[var].isel(time=time_idx), var, ax=ax[1], vmin=vmin, vmax=vmax)

        ax[0].set_title("Emulation")
        ax[1].set_title("Baseline")
        plt.tight_layout()
        plt.show()


pvars = ["cloud_water_mixing_ratio", "specific_humidity"]
target_prediction_2panel_meridional(prog, phys_baseline, pvars, time_idx=0)
target_prediction_2panel_meridional(prog, phys_baseline, pvars, time_idx=163)


# ## State Field Comparison
def select_time_vert(da, tidx, z=78, z_soil=0):
    da = da.isel(time=tidx)
    if "z" in da.dims:
        sfc_emu = da.isel(z=z)
    elif "z_soil" in da.dims:
        sfc_emu = da.isel(z_soil=z_soil)
    else:
        sfc_emu = da
        
    return sfc_emu


def get_vmin_vmax(var, da):
    
    if "mixing_ratio" in var:
        threshold = 98
    else:
        threshold = 99.5
    vmax = np.percentile(da, threshold)
    if np.any(da.values < 0):
        vmin = -vmax
    else:
        vmin = np.percentile(da, 100-threshold)
        
    return vmin, vmax


def plot_spatial_2panel_with_diff(emu, base, time_idx=0, level=75):
    skip_vars = [
        "latitude", "longitude", "x_wind", "y_wind", "land_sea_mask",
        "vertical_thickness_of_atmospheric_layer", "surface_geopotential"
    ]
    for var, da in emu.items():
        if var in skip_vars:
            continue
        base_var = var
        if base_var not in base:
            print(f"{base_var} missing from baseline inputs")
            continue

        fig = plt.figure()
        ax = fig.add_subplot(131, projection=ccrs.Robinson())
        ax2 = fig.add_subplot(132, projection=ccrs.Robinson())
        ax3 = fig.add_subplot(133, projection=ccrs.Robinson())
        fig.set_size_inches(15, 4)
        fig.set_dpi(80)

        emu_layer = select_time_vert(da, time_idx, z=level)
        vmin, vmax = get_vmin_vmax(var, emu_layer)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        
        plot_kw = dict(ax=ax, vmin=vmin, vmax=vmax)
        plot_var(emu_layer.to_dataset(name=var), var, plot_kwargs=dict(ax=ax, vmin=vmin, vmax=vmax))
        
        phy_layer = select_time_vert(base[base_var], time_idx, z=level)
        plot_kw["ax"] = ax2
        plot_var(phy_layer.to_dataset(name=var), var, plot_kwargs=dict(ax=ax2, vmin=vmin, vmax=vmax))

        diff = emu_layer - phy_layer
        vmin, vmax = get_vmin_vmax(var, diff)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        plot_var(diff.to_dataset(name=var), var, plot_kwargs=dict(ax=ax3, vmin=vmin, vmax=vmax))

        ax.set_title(f"Emulation: {var}")
        ax2.set_title("Baseline")
        ax3.set_title("Diff: Emu - Baseline")
        plt.show()


# ### Final output time (surface)
tidx = 160
prog.time.isel(time=tidx)
plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=78)


# ## Final output time (upper boundary layer)
plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=62)


# ## Final output time (upper atmosphere)
plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=34)

 
# ## Final output time (upper upper atmosphere)
plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=19)


# ## Drifts
drift_sel = {"3hr": slice(1, 2), "1day": slice(12, 16), "5day": slice(64, 80)}
drift_vars = ["cloud_water_mixing_ratio", "specific_humidity", "air_temperature"]
for drift, sel in drift_sel.items():
    if sel.stop >= len(prog.time):
        break
    
    p = prog_mean.isel(time=sel).mean(dim="time")
    b = base_mean.isel(time=sel).mean(dim="time")
    print((p - b)[drift_vars])


# ## Check spatial precip 
prog_w_diag["total_precipitation"]

# create a multi index coordinate to use xarray unstack
xy_points = list(product(np.arange(1, 49), repeat=2))
sample_coord = {"sample": pd.MultiIndex.from_tuples(xy_points, names=["y", "x"])}
diags = prog_w_diag.assign_coords(sample_coord).unstack("sample")

# precip start
plot_var(diags, "total_precipitation", isel_kwargs=dict(time=0))
plot_var(diags, "total_precipitation_physics_diag", isel_kwargs=dict(time=0))

# precip end
plot_var(diags, "total_precipitation", isel_kwargs=dict(time=-1))
plot_var(diags, "total_precipitation_physics_diag", isel_kwargs=dict(time=-1))


def split_emu_piggy_back(ds):
    
    diag_suffix = "_physics_diag"
    rain_wat = "tendency_of_rain_water_mixing_ratio"
    emu_keys = [k[:-len(diag_suffix)] for k in ds if diag_suffix in k]
    print(emu_keys)
    emu = ds[emu_keys]
    
    diags = {}
    for k, v in ds.items():
        if "_input" in k or "_output" in k:
            diags[k] = v
        elif diag_suffix in k:
            new_k = k[:-len(diag_suffix)]
            diags[new_k] = v
    diags = xr.Dataset(diags)
    diags = DerivedMapping(diags)
    as_ds = xr.Dataset({k: diags[k] for k in emu})
    
    return emu, as_ds


emu_del, diags_del = split_emu_piggy_back(diags)
plot_spatial_2panel_with_diff(emu_del, diags_del, time_idx=4)

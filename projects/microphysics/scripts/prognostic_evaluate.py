import argparse
import os
import fsspec
import logging
import hashlib
import tempfile
import wandb
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from itertools import product
from wandb.errors import CommError

from plot_utils import plot_var, plot_meridional
from fv3viz import infer_cmap_params
from fv3fit.tensorboard import plot_to_image
from vcm.catalog import catalog


logger = logging.getLogger(__name__)


# ## Global average time x height of SPHUM and T
def consistent_time_len(*da_args):
    # assumes same start time and time delta
    times = [len(da.time) for da in da_args]
    min_len = np.min(times)
    return [da.isel(time=slice(0, min_len)) for da in da_args]
    

def plot_time_vert_panels(da1, da2, dpi=80):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)
    fig.set_dpi(dpi)
    
    da1, da2 = consistent_time_len(da1, da2)
    vmin, vmax, cmap = infer_cmap_params(da2)
    vkw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
    da1.plot.pcolormesh(x="time", y="z", ax=ax[0], yincrease=False, **vkw)
    da2.plot.pcolormesh(x="time", y="z", ax=ax[1], yincrease=False, **vkw)
    (da1 - da2).plot.pcolormesh(x="time", y="z", ax=ax[2], yincrease=False)
    ax[0].set_title("Emulation")
    ax[1].set_title("Baseline")
    ax[2].set_title("Diff")
    
    for sub_ax in ax:
        sub_ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()

    return fig


# TODO: test this and maybe port back to diagnostics?
def _to_weighted_mean(da: xr.DataArray, area, dim):
    return da.weighted(area).mean(dim=dim)


def new_weighted_avg(ds, dim=["tile", "y", "x"]):

    area = ds["area"]
    area = area.fillna(0)
    weighted_mean = ds.map(_to_weighted_mean, keep_attrs=True, args=(area, dim))

    return weighted_mean


# TODO: use wandb to save preprocessed artifact for run that way it can be loaded if available
def get_avg_data(source_path: str, ds: xr.Dataset, run, filename="state_mean_by_height.nc", override_artifact=False):

    source_hash = hashlib.md5(source_path.encode("utf-8")).hexdigest()
    try:
        artifact = run.use_artifact(f"{source_hash}:latest")
        logger.info(f"Loaded existing artifact for: {source_path}")
    except CommError:
        logger.error(f"Run averaging artifact not found for: {source_path}")
        artifact = None

    if override_artifact or artifact is None:
        prog_avg = new_weighted_avg(ds)
        with ProgressBar():
            prog_avg.load()

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_file = os.path.join(tmpdir, filename)
            prog_avg.to_netcdf(nc_file)
            artifact = wandb.Artifact(source_hash, type="by_height_avg")
            artifact.add_file(nc_file)
            run.log_artifact(artifact)
    else:
        artifact_dir = artifact.download()
        prog_avg = xr.open_dataset(os.path.join(artifact_dir, filename))

    return prog_avg





# pvars = ["cloud_water_mixing_ratio", "specific_humidity"]
# target_prediction_2panel_meridional(prog, phys_baseline, pvars, time_idx=0)
# target_prediction_2panel_meridional(prog, phys_baseline, pvars, time_idx=163)


# # ## State Field Comparison
# def select_time_vert(da, tidx, z=78, z_soil=0):
#     da = da.isel(time=tidx)
#     if "z" in da.dims:
#         sfc_emu = da.isel(z=z)
#     elif "z_soil" in da.dims:
#         sfc_emu = da.isel(z_soil=z_soil)
#     else:
#         sfc_emu = da
        
#     return sfc_emu


# def get_vmin_vmax(var, da):
    
#     if "mixing_ratio" in var:
#         threshold = 98
#     else:
#         threshold = 99.5
#     vmax = np.percentile(da, threshold)
#     if np.any(da.values < 0):
#         vmin = -vmax
#     else:
#         vmin = np.percentile(da, 100-threshold)
        
#     return vmin, vmax


# def plot_spatial_2panel_with_diff(emu, base, time_idx=0, level=75):
#     skip_vars = [
#         "latitude", "longitude", "x_wind", "y_wind", "land_sea_mask",
#         "vertical_thickness_of_atmospheric_layer", "surface_geopotential"
#     ]
#     for var, da in emu.items():
#         if var in skip_vars:
#             continue
#         base_var = var
#         if base_var not in base:
#             print(f"{base_var} missing from baseline inputs")
#             continue

#         fig = plt.figure()
#         ax = fig.add_subplot(131, projection=ccrs.Robinson())
#         ax2 = fig.add_subplot(132, projection=ccrs.Robinson())
#         ax3 = fig.add_subplot(133, projection=ccrs.Robinson())
#         fig.set_size_inches(15, 4)
#         fig.set_dpi(80)

#         emu_layer = select_time_vert(da, time_idx, z=level)
#         vmin, vmax = get_vmin_vmax(var, emu_layer)
#         if vmin > vmax:
#             vmin, vmax = vmax, vmin
        
#         plot_kw = dict(ax=ax, vmin=vmin, vmax=vmax)
#         plot_var(emu_layer.to_dataset(name=var), var, plot_kwargs=dict(ax=ax, vmin=vmin, vmax=vmax))
        
#         phy_layer = select_time_vert(base[base_var], time_idx, z=level)
#         plot_kw["ax"] = ax2
#         plot_var(phy_layer.to_dataset(name=var), var, plot_kwargs=dict(ax=ax2, vmin=vmin, vmax=vmax))

#         diff = emu_layer - phy_layer
#         vmin, vmax = get_vmin_vmax(var, diff)
#         if vmin > vmax:
#             vmin, vmax = vmax, vmin
#         plot_var(diff.to_dataset(name=var), var, plot_kwargs=dict(ax=ax3, vmin=vmin, vmax=vmax))

#         ax.set_title(f"Emulation: {var}")
#         ax2.set_title("Baseline")
#         ax3.set_title("Diff: Emu - Baseline")
#         plt.show()


# ### Final output time (surface)
# tidx = 160
# prog.time.isel(time=tidx)
# plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=78)


# # ## Final output time (upper boundary layer)
# plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=62)


# # ## Final output time (upper atmosphere)
# plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=34)

 
# # ## Final output time (upper upper atmosphere)
# plot_spatial_2panel_with_diff(prog, phys_baseline, time_idx=tidx, level=19)


# ## Drifts
# drift_sel = {"3hr": slice(1, 2), "1day": slice(12, 16), "5day": slice(64, 80)}
# drift_vars = ["cloud_water_mixing_ratio", "specific_humidity", "air_temperature"]
# for drift, sel in drift_sel.items():
#     if sel.stop >= len(prog.time):
#         break
    
#     p = prog_mean.isel(time=sel).mean(dim="time")
#     b = base_mean.isel(time=sel).mean(dim="time")
#     print((p - b)[drift_vars])


# ## Check spatial precip 
# prog_w_diag["total_precipitation"]

# # create a multi index coordinate to use xarray unstack
# xy_points = list(product(np.arange(1, 49), repeat=2))
# sample_coord = {"sample": pd.MultiIndex.from_tuples(xy_points, names=["y", "x"])}
# diags = prog_w_diag.assign_coords(sample_coord).unstack("sample")

# # precip start
# plot_var(diags, "total_precipitation", isel_kwargs=dict(time=0))
# plot_var(diags, "total_precipitation_physics_diag", isel_kwargs=dict(time=0))

# # precip end
# plot_var(diags, "total_precipitation", isel_kwargs=dict(time=-1))
# plot_var(diags, "total_precipitation_physics_diag", isel_kwargs=dict(time=-1))


# def split_emu_piggy_back(ds):
    
#     diag_suffix = "_physics_diag"
#     rain_wat = "tendency_of_rain_water_mixing_ratio"
#     emu_keys = [k[:-len(diag_suffix)] for k in ds if diag_suffix in k]
#     print(emu_keys)
#     emu = ds[emu_keys]
    
#     diags = {}
#     for k, v in ds.items():
#         if "_input" in k or "_output" in k:
#             diags[k] = v
#         elif diag_suffix in k:
#             new_k = k[:-len(diag_suffix)]
#             diags[new_k] = v
#     diags = xr.Dataset(diags)
#     diags = DerivedMapping(diags)
#     as_ds = xr.Dataset({k: diags[k] for k in emu})
    
#     return emu, as_ds


# emu_del, diags_del = split_emu_piggy_back(diags)
# plot_spatial_2panel_with_diff(emu_del, diags_del, time_idx=4)


def plot_time_by_heights(prognostic, baseline):
    
    plot_time_height_vars = [
        "cloud_water_mixing_ratio",
        "specific_humidity",
        "air_temperature",
        "eastward_wind",
        "northward_wind",
    ]

    for name in plot_time_height_vars:
        fig = plot_time_vert_panels(
            prognostic[name],
            baseline[name]
        )
        wandb.log({f"avg_time_height/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def plot_global_means(prognostic, baseline):

    prognostic, baseline = consistent_time_len(prognostic, baseline)

    for varname, da in prognostic.items():
        if varname in ["land_sea_mask", "latitude", "longitude", "area"]:
            continue
        if varname not in baseline:
            logger.info(f"Skipping global mean due to missing basline variable: {varname}")
            continue

        fig, ax = plt.subplots()
        fig.set_dpi(80)

        da.plot(ax=ax, label="Emulation")
        baseline[varname].plot(ax=ax, label="Baseline", alpha=0.6)
        plt.legend()

        wandb.log({f"global_avg/{varname}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def plot_transects(prognostic, baseline):

    varnames = ["cloud_water_mixing_ratio", "specific_humidity", "air_temperature"]

    tidx_map = {"start": 0, "near_end": len(prognostic.time) - 2}

    for time_name, tidx in tidx_map.items():
        for name in varnames:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(8, 4)
            fig.set_dpi(80)

            plot_meridional(prognostic.isel(time=tidx), name, ax=ax[0])
            plot_meridional(baseline.isel(time=tidx), name, ax=ax[1])

            ax[0].set_title("Emulation")
            ax[1].set_title("Baseline")
            plt.tight_layout()

            log_name = f"meridional_transect/{time_name}/{name}"
            wandb.log({log_name: wandb.Image(plot_to_image(fig))})
            plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("prognostic_path", help="Path or url to a prognostic run")
    parser.add_argument(
        "--baseline-path",
        help="Path or url to a baseline run for comparison",
        default=(
            "gs://vcm-ml-experiments/andrep/2021-05-28/"
            "spunup-baseline-simple-phys-hybrid-edmf-extended/fv3gfs_run"
        )
    )
    parser.add_argument(
        "--grid-key",
        default="c48",
        help="Grid to load from catalog for area-weighted averages"
    )
    parser.add_argument(
        "--override-artifacts",
        action="store_true",
        help="Force upload of the averaging artifacts"
    )
    parser.add_argument(
        "--wandb-project",
        default="scratch-project",
    )
    
    args = parser.parse_args()
    run = wandb.init(job_type="prognostic_evaluation", entity="ai2cm", project=args.wandb_project)

    # TODO: Prognostic runs should produce a useable artifact directing to output?

    wandb.config.update(args)

    path = args.prognostic_path
    baseline_path = args.baseline_path
    prog = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "state_after_timestep.zarr")), consolidated=True)
    baseline = xr.open_zarr(fsspec.get_mapper(os.path.join(baseline_path, "state_after_timestep.zarr")), consolidated=True)

    grid = catalog[f"grid/{args.grid_key}"].to_dask()
    prog = prog.merge(grid).isel(time=slice(0, 8))
    baseline = baseline.merge(grid).isel(time=slice(0, 8))

    prog_mean_by_height = get_avg_data(path, prog, run, override_artifact=args.override_artifacts)
    base_mean_by_height = get_avg_data(baseline_path, baseline, run, override_artifact=args.override_artifacts)

    plot_time_by_heights(prog_mean_by_height, base_mean_by_height)

    ## Global average comparison after timestep
    prog_mean = prog_mean_by_height.mean(dim=["z"]).compute()
    base_mean = base_mean_by_height.mean(dim=["z", "z_soil"]).compute()

    plot_global_means(prog_mean, base_mean)

    # Some meridional transects
    plot_transects(prog, baseline)



    # TODO: Maybe remove?
    # prog_w_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "state_output.zarr")))
    # prog_sfc_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(path, "sfc_dt_atmos.zarr")), consolidated=True)
    # base_diag = xr.open_zarr(fsspec.get_mapper(os.path.join(baseline_path, "sfc_dt_atmos.zarr")), consolidated=True)


if __name__ == "__main__":
    
    # path = f"gs://vcm-ml-scratch/andrep/2021-10-02-wandb-training/prognostic-runs/all-tends-limited-dense"
    main()

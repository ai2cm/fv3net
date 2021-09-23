import xarray as xr
import os
import vcm
import vcm.fv3.metadata
import vcm.catalog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datapane
import wandb
import sys
import subprocess
import config

import argparse
import json

PLOTS = []
PAGES = []
current_page = "Intro"

matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.dpi"] = 100


def add_plot(fig, label=None):
    PLOTS.append(datapane.Plot(fig, label=label, caption=label))


def start_page(name):
    global current_page

    current_page = name


def finish_page():
    PAGES.append(datapane.Page(blocks=PLOTS, title=current_page))
    PLOTS.clear()


def weighted_average(x, weight, dims=None):
    dims = dims or weight.dims
    out = (x * weight).sum(weight.dims) / weight.sum(dims)

    try:
        return out.rename(x.name)
    except AttributeError:
        return out


def is_sea(land_sea_mask):
    return xr.where(vcm.xarray_utils.isclose(land_sea_mask, 0), 1.0, 0.0)


def is_land(land_sea_mask):
    return xr.where(vcm.xarray_utils.isclose(land_sea_mask, 1), 1.0, 0.0)


def open_run(local_path, data="diags_3d.zarr"):

    ds = xr.open_zarr(os.path.join(local_path, data))
    # get tropical ocean averaged forcing

    if "grid_xt" in ds.dims:
        ds = vcm.fv3.metadata.gfdl_to_standard(ds)
    grid = vcm.catalog.catalog["grid/c48"].to_dask()
    lsm = vcm.catalog.catalog["landseamask/c48"].to_dask()
    return grid.merge(ds, compat="override").merge(lsm)


def compute_masks(ds):
    return {
        "global": True,
        "land": is_land(ds.land_sea_mask),
        "ocean": is_sea(ds.land_sea_mask),
        "tropical ocean": is_sea(ds.land_sea_mask) * (np.abs(ds.lat) < 10),
    }


def regional_average(ds):
    """Average a dataset over a given region"""
    masks = compute_masks(ds)
    averages = []

    for (name, mask) in masks.items():
        averages.append(
            weighted_average(ds, ds.area.where(mask, 0)).assign_coords(region=name)
        )
    return xr.concat(averages, dim="region")


def average_data(ds, field="specific_humidity"):
    masks = compute_masks(ds)
    fig, axs = plt.subplots(len(masks), 3, figsize=(14, 14), constrained_layout=True)

    for i, (name, mask) in enumerate(masks.items()):
        for j, tend in enumerate(["emulator", "fv3_physics", "dynamics"]):
            avg = weighted_average(
                ds[f"tendency_of_{field}_due_to_{tend}"], ds.area.where(mask, 0)
            )
            avg.plot(ax=axs[i, j], y="z", yincrease=False, rasterized=True)
            axs[i, j].set_title(f"{name} {tend}")

    return fig


def plot_field(path, label, field="PWAT"):
    ds = xr.merge(
        [
            open_run(path, data="atmos_dt_atmos.zarr"),
            open_run(path, data="sfc_dt_atmos.zarr"),
        ],
        compat="override",
    )
    masks = compute_masks(ds)
    weighted_average(ds, masks["tropical ocean"])[field].plot(label=label)
    plt.title(ds[field].long_name)


def plot_offline_online_comparison_tropical_ocean(field):
    plt.figure()
    plot_field(online_url, "online 2wax95yx", field)
    plot_field(offline_url, "offline 3479nauv", field)
    plt.legend()

    return plt.gcf()


start_page("Tropical ocean averages")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("offline_id")
    parser.add_argument("online_id")

    args = parser.parse_args()

    offline_id = args.offline_id
    online_id = args.online_id

    offline_url = f"/scratch/runs/prognostic-runs/{offline_id}/"
    online_url = f"/scratch/runs/prognostic-runs/{online_id}/"

    offline_link = (f"https://wandb.ai/ai2cm/emulator-noah/runs/{offline_id}",)
    online_link = (f"https://wandb.ai/ai2cm/emulator-noah/runs/{online_id}",)
    PAGES.append(
        datapane.Page(
            datapane.Text(
                json.dumps(
                    {
                        "offline run": offline_link,
                        "online run": online_link,
                        "args": " ".join(sys.argv),
                        "git rev": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"]
                        ).decode(),
                    }
                )
            ),
            title="Metadata",
        ),
    )

    start_page("ROIs")
    add_plot(average_data(open_run(offline_url)), "offline ROIs")
    add_plot(average_data(open_run(online_url)), "online ROIs")
    finish_page()

    for field in config.tropical_averages:
        add_plot(plot_offline_online_comparison_tropical_ocean(field), label=field)

    # %%
    ds_online = open_run(online_url, "sfc_dt_atmos.zarr").merge(
        open_run(online_url, "diags.zarr")
    )

    url = offline_url
    ds_offline = open_run(url, "sfc_dt_atmos.zarr").merge(open_run(url, "diags.zarr"))

    region = "tropical ocean"
    avg_online = weighted_average(ds_online, compute_masks(ds_online)[region])
    avg_offline = weighted_average(ds_offline, compute_masks(ds_online)[region])

    relative_change = (avg_online - avg_offline).isel(time=-1) / (
        avg_offline.isel(time=0)
    )

    changes = (
        relative_change.drop(["latb", "lonb", "x_interface", "y_interface"])
        .to_array()
        .to_series()
    )
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    changes.sort_values().plot(kind="bar", ax=ax)
    add_plot(fig, label="Variables with biggest offline online difference")
    finish_page()

    def plot_all_piggies(field, region="tropical ocean"):
        avg_online = weighted_average(ds_online, compute_masks(ds_online)[region])
        avg_offline = weighted_average(ds_offline, compute_masks(ds_online)[region])
        plt.figure()
        avg_online[f"{field}_due_to_emulator"].plot(label="emulator online", c="b")
        avg_online[f"{field}_due_to_fv3_physics"].plot(
            label="fv3 physics online", ls="--", c="b"
        )
        avg_offline[f"{field}_due_to_emulator"].plot(label="emulator offline", c="g")
        avg_offline[f"{field}_due_to_fv3_physics"].plot(
            label="fv3 physics offline", c="g", ls="--"
        )
        plt.title(field + " " + region)
        plt.legend()

        return plt.gcf()

    # start_page("Piggy backs")

    # add_plot(plot_all_piggies(field="storage_of_eastward_wind_path"))
    # add_plot(plot_all_piggies("storage_of_air_temperature_path"))
    # add_plot(plot_all_piggies("storage_of_air_temperature_path", region="land"))
    # add_plot(plot_all_piggies("storage_of_specific_humidity_path"))
    # finish_page()

    job = wandb.init(job_type="3hour-eval", project="emulator-noah", entity="ai2cm")
    datapane.Report(blocks=PAGES).save("report.html")
    wandb.log({"3 hour report": wandb.Html(open("report.html"))})

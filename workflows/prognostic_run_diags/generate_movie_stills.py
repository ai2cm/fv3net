import argparse
import logging
import os
from pathlib import Path
from multiprocessing import Pool

import intake
import xarray as xr
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import vcm
import save_prognostic_run_diags as save_diags
import load_diagnostic_data as load_diags


def _open_zarr(url):
    m = fsspec.get_mapper(url)
    return xr.open_zarr(m, consolidated=True)

def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    return str(TOP_LEVEL_DIR / "catalog.yml")


def _compute_q_vars(ds):
    arrays = []
    for func in [
        save_diags._column_pq1,
        save_diags._column_pq2,
        save_diags._column_dq1,
        save_diags._column_dq2,
        save_diags._column_q1,
        save_diags._column_q2,
    ]:
        arrays.append(func(ds))
    return xr.merge(arrays)


plot_lims = {
    'column_integrated_pQ1': (-600, 600),
    'column_integrated_dQ1': (-600, 600),
    'column_integrated_Q1': (-600, 600),
    'column_integrated_pQ2': (-20, 20),
    'column_integrated_dQ2': (-20, 20),
    'column_integrated_Q2': (-20, 20),
}

COORD_NAMES = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
}

_COORD_VARS = {
    "lonb": ["y_interface", "x_interface", "tile"],
    "latb": ["y_interface", "x_interface", "tile"],
    "lon": ["y", "x", "tile"],
    "lat": ["y", "x", "tile"],
}

SUBPLOT_KW = {"projection": ccrs.Robinson()}


def _six_panel_heating_moistening(ds, axes):
    i = 0
    for var, vlims in plot_lims.items():
        ax = axes.flatten()[i]
        mv = vcm.mappable_var(ds, var, coord_vars=_COORD_VARS, **COORD_NAMES)
        vcm.plot_cube(mv, ax=ax, vmin=vlims[0], vmax=vlims[1], cmap='RdBu_r')
        ax.set_title(var.replace("_", " "))
        i += 1


def _plot_and_save(t):
    fig_filename = f"heating_and_moistening_{t:05}.png"
    plotme = plot_vars.isel(time=t)
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.3), subplot_kw=SUBPLOT_KW)
    _six_panel_heating_moistening(plotme, axes)
    fig.suptitle(plotme.time.values.item())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    with fsspec.open(os.path.join(args.output, fig_filename), "wb") as fig_file:
        fig.savefig(fig_file, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    CATALOG = _catalog()

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Path to rundir")
    parser.add_argument("output", help="Output location for movie stills")
    args = parser.parse_args()

    if vcm.cloud.get_protocol(args.output) == "file":
        os.makedirs(args.output, exist_ok=True)

    catalog = intake.open_catalog(CATALOG)

    grid = catalog['grid/c48'].to_dask().load()
    ds = load_diags._load_prognostic_run_physics_output(args.url)
    plot_vars = _compute_q_vars(ds)
    plot_vars = plot_vars.merge(grid)
    T = plot_vars.sizes["time"]
    logger.info(f"Saving {T} still images to {args.output}")
    with Pool(8) as p:
        p.map(_plot_and_save, range(T))



import argparse
import fsspec
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Mapping, Sequence
import xarray as xr


DERIVATION_DIM = "derivation"
DOMAIN_DIM = "domain"
UNITS_Q1 = "K/s"
UNITS_Q2 = "kg/kg/s"

# grid info for the plot_cube function
MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}

GLOBAL_MEAN_VARS = [
    'column_integrated_dQ1_global_mean', 
    'column_integrated_pQ1_global_mean', 
    'column_integrated_Q1_global_mean',
    'column_integrated_dQ2_global_mean', 
    'column_integrated_pQ2_global_mean', 
    'column_integrated_Q2_global_mean']

PROFILE_VARS = ['dQ1', 'pQ1', 'Q1', 'dQ2', 'pQ2', 'Q2']


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags_report")


def _plot_diurnal_multivariable(
        ds: xr.Dataset,
        diurnal_vars: Sequence[str],
        x_dim: str,
        title: str,
        units: str = None,
        derivation_dim: str = DERIVATION_DIM,
        selection: Mapping[str, str] = None
):
    fig = plt.figure()
    time_coords = ds[diurnal_vars[0]][x_dim].values
    selection = selection or {}
    for diurnal_var in diurnal_vars:
        for derivation_coord in ds[derivation_dim].values:
            plt.plot(
                time_coords,
                ds[diurnal_var].sel(selection),
                label=f"{diurnal_var}, {derivation}".replace("_", " ")
                )
    plt.xlabel("local time [hr]")
    plt.ylabel(units or "")
    plt.legend()
    fig.savefig(f'{title.replace(" ", "_")}.png')


def _plot_profile_vars(
        ds: xr.Dataset,
        dpi: int = 100,
        profile_vars: Sequence[str] = PROFILE_VARS,
        derivation_dim: str = DERIVATION_DIM,
        domain_dim: str = DOMAIN_DIM,
        units_q1: str = UNITS_Q1,
        units_q2: str = UNITS_Q2,
        ):
    for var in profile_vars:
        facet_grid = ds[var].plot(y='z', hue=derivation_dim, col=domain_dim)
        facet_grid.set_titles(template="{value}", maxchar=40)
        f = facet_grid.fig
        for ax in facet_grid.axes.flatten():
            ax.invert_yaxis()
            ax.plot([0, 0], [1, 79], 'k-')
            if '1' in var:
                ax.set_xlim([-0.0001, 0.0001])
                ax.set_xticks(np.arange(-1e-4, 1.1e-4, 5e-5))
                ax.set_xlabel(f"{var} {[units_q1]}")
            else:
                ax.set_xlim([-1e-7, 1e-7])
                ax.set_xticks(np.arange(-1e-7, 1.1e-7, 5e-8))
                ax.set_xlabel(f"{var} {[units_q2]}")
        f.set_size_inches([17,3.5])
        f.set_dpi(dpi)
        f.suptitle(var)
        f.savefig(f'{var}_profile_plot.png')
        

def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets_file",
        type=str,
        help=("Json with runs to compare and their paths."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    return parser.parse_args()


if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()
    
    with open(args.datasets_file, "r") as f:
        runs = json.load(f)
    data = {}
    for label, path in runs.items():
        with fsspec.open(path, "rb") as f:
            data[label] = xr.open_dataset(f).load()
    
    


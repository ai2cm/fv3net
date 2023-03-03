from typing import Hashable, List, Mapping
import xarray as xr
import numpy as np
import os

output_variables = [
    "T_mid",
    "qv",
    "omega",
    "horiz_winds",  # this will be changed to u and v in the future
    "surf_sens_flux",
    "surf_evap",
    "SW_clrsky_flux_up@bot",
    "SW_clrsky_flux_dn@bot",
    "LW_clrsky_flux_dn@bot",
    "LW_clrsky_flux_up@bot",
    "SW_clrsky_flux_up@tom",
    "LW_clrsky_flux_up@tom",
    "VerticalLayerInterface",
    # "pseudo_density",  # this is pressure thickness
    "area",
    "lat",
    "lon",
    "hyai",
    "hybi",
]

NP = 4
NPG = 2
levs = 128


def convert_to_fv3_format(
    NE: int,
    data_path: str,
    file_name: str,
    output_path: str,
    output_file_name: str,
    output_variables: List[str],
):
    """Converts a scream output file to the fv3gfs format.

    Args:
        NE: number of spectral elements per cubed face
        data_path
        file_name
        output_path
        output_file_name
        output_variables
    """
    Ncols = NE ** 2 * 6 * NPG ** 2
    x_dim = int(Ncols / 6 / NE / 3)
    y_dim = int(Ncols / 6 / x_dim)
    ds = xr.open_mfdataset(
        os.path.join(data_path, file_name),
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
    )
    if "horiz_winds" in output_variables:
        u = ds.horiz_winds.isel(dim2=0).rename({"eastward_wind"})
        v = ds.horiz_winds.isel(dim2=1).rename({"northward_wind"})

    rename_vars: Mapping[Hashable, Hashable] = {
        "T_mid": "air_temperature",
        "qv": "specific_humidity",
        "omega": "upward_air_velocity",
        "surf_sens_flux": "sensible_heat_flux",
        "surf_evap": "latent_heat_flux",
        "SW_clrsky_flux_up@bot": "clear_sky_upward_shortwave_flux_at_surface",
        "SW_clrsky_flux_dn@bot": "clear_sky_downward_shortwave_flux_at_surface",
        "LW_clrsky_flux_dn@bot": "clear_sky_downward_longwave_flux_at_surface",
        "LW_clrsky_flux_up@bot": "clear_sky_upward_longwave_flux_at_surface",
        "SW_clrsky_flux_up@tom": "clear_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "LW_clrsky_flux_up@tom": "clear_sky_upward_longwave_flux_at_top_of_atmosphere",
        "VerticalLayerInterface": "vertical_thickness_of_atmospheric_layer",
        # "pseudo_density": "pressure_thickness_of_atmospheric_layer",
    }
    ds = ds[output_variables].rename(rename_vars)
    ds = ds.drop("horiz_winds")
    ds["eastward_wind"] = u
    ds["northward_wind"] = v
    ds = ds.rename({"lev": "z", "ilev": "z_interface"})
    ds = ds.assign_coords(
        {
            "z": np.arange(0, levs),
            "z_interface": np.arange(0, levs + 1),
            "ncol": np.arange(0, Ncols),
            "tile": np.arange(0, 6),
            "x": np.arange(0, x_dim),
            "y": np.arange(0, y_dim),
        }
    )
    ds = ds.stack(ncol=("tile", "x", "y"))
    ds = ds.unstack("ncol")
    ds = ds.transpose(..., "tile", "z", "z_interface", "y", "x")
    ds.to_zarr(os.path.join(output_path, output_file_name), consolidated=True)

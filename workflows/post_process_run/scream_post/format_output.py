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
    "p_mid",
    "ps",
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


def make_placeholder_data(
    sample: xr.DataArray, generate_variable: str, scaled_factor: float = 1.0
):
    placeholder = sample.copy()
    placeholder.values[:] = np.random.rand(*placeholder.values.shape) * scaled_factor
    placeholder.attrs = {}
    return placeholder.rename(generate_variable)


def make_delp(p_mid, z_int, ps):
    ptop = 225.52
    delp = p_mid.copy()
    dz = z_int[:, :, 0:-1] - z_int[:, :, 1:]
    p_int = z_int.copy()
    p_int[:, :, 0] = ptop
    for k in range(1, levs):
        p_int[:, :, k] = (
            dz[:, :, k - 1] * p_mid[:, :, k] + dz[:, :, k] * p_mid[:, :, k - 1]
        ) / (dz[:, :, k - 1] + dz[:, :, k])
    p_int[:, :, -1] = ps / 2.0  # surface pressure bug
    delp = p_int[:, :, 1:] - p_int[:, :, 0:-1]
    return delp.rename({"ilev": "lev"}).rename(
        "pressure_thickness_of_atmospheric_layer"
    )


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
        data_path: path to input data directory, remote or local
        file_name: either a single file name
            or a wildcard pattern to match multiple names
        output_path: path to output data directory, remote or local
        output_file_name: name of output file
        output_variables: list of variables to output
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
    ds["pressure_thickness_of_atmospheric_layer"] = make_delp(
        ds.p_mid, ds.vertical_thickness_of_atmospheric_layer, ds.ps
    )

    ds["air_temperature_tendency_due_to_nudging"] = make_placeholder_data(
        ds.air_temperature, "air_temperature_tendency_due_to_nudging", 1e-1
    )
    ds["specific_humidity_tendency_due_to_nudging"] = make_placeholder_data(
        ds.air_temperature, "specific_humidity_tendency_due_to_nudging", 1e-5
    )
    ds["x_wind_tendency_due_to_nudging"] = make_placeholder_data(
        ds.air_temperature, "x_wind_tendency_due_to_nudging", 1e-3
    )
    ds["y_wind_tendency_due_to_nudging"] = make_placeholder_data(
        ds.air_temperature, "y_wind_tendency_due_to_nudging", 1e-3
    )
    ds["tendency_of_air_temperature_due_to_scream_physics"] = make_placeholder_data(
        ds.air_temperature, "tendency_of_air_temperature_due_to_scream_physics", 1e-3
    )
    ds["tendency_of_specific_humidity_due_to_scream_physics"] = make_placeholder_data(
        ds.air_temperature, "tendency_of_specific_humidity_due_to_scream_physics", 1e-6
    )
    ds["tendency_of_eastward_wind_due_to_scream_physics"] = make_placeholder_data(
        ds.air_temperature, "tendency_of_eastward_wind_due_to_scream_physics", 1e-4
    )
    ds["tendency_of_northward_wind_due_to_scream_physics"] = make_placeholder_data(
        ds.air_temperature, "tendency_of_northward_wind_due_to_scream_physics", 1e-4
    )

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

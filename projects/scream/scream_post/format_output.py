from typing import List
import xarray as xr
import numpy as np
import os
import cftime

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


def make_placeholder_data(
    sample: xr.DataArray, generate_variable: str, scaled_factor: float = 1.0
):
    placeholder = xr.DataArray(
        np.random.rand(*sample.values.shape) * scaled_factor,
        coords=sample.coords,
        dims=sample.dims,
    )
    return placeholder.rename(generate_variable)


def make_delp(ps, hyai, hybi):
    p0 = 100000.0
    ps = (
        ps / 2.0
    )  # this round of ps is 2x the actual ps, remove this once we have the updated data
    p_int = p0 * hyai + ps * hybi
    delp = (
        p_int.diff("ilev")
        .rename({"ilev": "lev"})
        .rename("pressure_thickness_of_atmospheric_layer")
    )
    return delp


def make_new_date(start_date: cftime.DatetimeGregorian, n: int):
    new_date = []
    for i in range(n):
        new_date.append(
            cftime.DatetimeJulian(
                2016,
                start_date.month + i,
                start_date.day,
                start_date.hour,
                start_date.minute,
            )
        )

    return new_date


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
    ds = xr.open_mfdataset(
        os.path.join(data_path, file_name),
        concat_dim="time",
        combine="nested",
        data_vars="minimal",
    )
    if "horiz_winds" in output_variables:
        u = ds.horiz_winds.isel(dim2=0).rename({"x_wind"})
        v = ds.horiz_winds.isel(dim2=1).rename({"y_wind"})

    ds = ds[output_variables]
    ds = ds.drop("horiz_winds")
    ds["x_wind"] = u
    ds["y_wind"] = v
    ds["pressure_thickness_of_atmospheric_layer"] = make_delp(ds.ps, ds.hyai, ds.hybi)

    ds[
        "pressure_thickness_of_atmospheric_layer_tendency_due_to_nudging"
    ] = make_placeholder_data(
        ds.T_mid,
        "pressure_thickness_of_atmospheric_layer_tendency_due_to_nudging",
        1e-3,
    )
    ds["T_mid_tendency_due_to_nudging"] = make_placeholder_data(
        ds.T_mid, "T_mid_tendency_due_to_nudging", 1e-3
    )
    ds["qv_tendency_due_to_nudging"] = make_placeholder_data(
        ds.T_mid, "qv_tendency_due_to_nudging", 1e-7
    )
    ds["x_wind_tendency_due_to_nudging"] = make_placeholder_data(
        ds.T_mid, "x_wind_tendency_due_to_nudging", 1e-3
    )
    ds["y_wind_tendency_due_to_nudging"] = make_placeholder_data(
        ds.T_mid, "y_wind_tendency_due_to_nudging", 1e-3
    )
    ds["tendency_of_T_mid_due_to_scream_physics"] = make_placeholder_data(
        ds.T_mid, "tendency_of_T_mid_due_to_scream_physics", 1e-5
    )
    ds["tendency_of_qv_due_to_scream_physics"] = make_placeholder_data(
        ds.T_mid, "tendency_of_qv_due_to_scream_physics", 1e-8
    )
    ds["tendency_of_eastward_wind_due_to_scream_physics"] = make_placeholder_data(
        ds.T_mid, "tendency_of_eastward_wind_due_to_scream_physics", 1e-5
    )
    ds["tendency_of_northward_wind_due_to_scream_physics"] = make_placeholder_data(
        ds.T_mid, "tendency_of_northward_wind_due_to_scream_physics", 1e-5
    )
    # make new date here because this round of data has year 0001
    # and causes issues with diagnostics code
    new_date = make_new_date(ds.time.values[0], len(ds.time))
    ds = ds.assign_coords(time=new_date)
    ds = ds.rename({"lev": "z", "ilev": "z_interface"})

    nudging_variables = [
        "T_mid_tendency_due_to_nudging",
        "qv_tendency_due_to_nudging",
        "x_wind_tendency_due_to_nudging",
        "y_wind_tendency_due_to_nudging",
    ]
    ds[nudging_variables].to_zarr(
        os.path.join(output_path, "nudging_tendencies.zarr"), consolidated=True
    )
    physics_tendencies = [
        "tendency_of_T_mid_due_to_scream_physics",
        "tendency_of_qv_due_to_scream_physics",
        "tendency_of_eastward_wind_due_to_scream_physics",
        "tendency_of_northward_wind_due_to_scream_physics",
    ]
    ds[physics_tendencies].to_zarr(
        os.path.join(output_path, "physics_tendencies.zarr"), consolidated=True
    )
    ds.drop(nudging_variables).drop(physics_tendencies).to_zarr(
        os.path.join(output_path, output_file_name), consolidated=True
    )

import xarray as xr
import numpy as np
import pandas as pd
import cftime
import logging


logger = logging.getLogger(__name__)


def convert_npdatetime_to_cftime(ds: xr.Dataset):
    if isinstance(ds.time.values[0], np.datetime64):
        cf_time = []
        for date in ds.time.values:
            date = pd.to_datetime(date)
            cf_time.append(
                cftime.DatetimeJulian(
                    date.year,
                    date.month,
                    date.day,
                    date.hour,
                    date.minute,
                    date.second,
                )
            )
        ds["time"] = xr.DataArray(cf_time, coords=ds.time.coords, attrs=ds.time.attrs)
    return ds


def rename_lev_to_z(ds: xr.Dataset):
    rename_vars = {"lev": "z", "ilev": "z_interface"}
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds.dims}
    return ds.rename(rename_vars)


def split_horiz_winds_tend(ds: xr.Dataset, label: str):
    u = ds[f"{label}_horiz_winds_tend"].isel(dim2=0).rename({f"{label}_U_tend"})
    v = ds[f"{label}_horiz_winds_tend"].isel(dim2=1).rename({f"{label}_V_tend"})
    ds = ds.drop(f"{label}_horiz_winds_tend")
    ds[f"{label}_U_tend"] = u
    ds[f"{label}_V_tend"] = v
    return ds


def add_rad_fluxes(ds: xr.Dataset):
    if (
        "SW_flux_dn_at_model_bot"
        and "SW_flux_dn_at_model_top"
        and "LW_flux_dn_at_model_bot" in ds.variables
    ):
        DSWRFsfc = ds.SW_flux_dn_at_model_bot
        DSWRFtoa = ds.SW_flux_dn_at_model_top
        DLWRFsfc = ds.LW_flux_dn_at_model_bot
    elif "DSWRFsfc" and "DSWRFtoa" and "DLWRFsfc" in ds.variables:
        DSWRFsfc = ds.DSWRFsfc
        DSWRFtoa = ds.DSWRFtoa
        DLWRFsfc = ds.DLWRFsfc
    else:
        logger.warning(
            "No radiation fluxes found in dataset. Skipping radiation calculations."
        )
        return ds

    shortwave_transmissivity_of_atmospheric_column = DSWRFsfc / DSWRFtoa
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.where(  # noqa
        DSWRFtoa != 0.0, 0.0
    )
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.assign_attrs(  # noqa
        units="-", long_name="shortwave transmissivity of atmospheric column"
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = (  # noqa
        DLWRFsfc
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface.assign_attrs(  # noqa
        units="W/m**2", long_name="surface downward longwave flux"
    )
    ds[
        "shortwave_transmissivity_of_atmospheric_column"
    ] = shortwave_transmissivity_of_atmospheric_column
    ds[
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
    ] = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface
    return ds

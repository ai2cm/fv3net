import xarray as xr
import numpy as np
import pandas as pd
import cftime


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

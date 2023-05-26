import datetime

import cftime
import dask.diagnostics
import xarray as xr


SST_CLIMATOLOGY_ZARR = (
    "gs://vcm-ml-intermediate/2023-05-26-RTGSST.1982.2012.monthly.clim.zarr"
)


def reindex_like_months(ds, times):
    """Input dataset must contain a single value per month.  Order is not
    important.
    """
    reindexed = ds.reindex(time=times)
    for month in range(1, 13):
        reindexed = xr.where(
            reindexed.time.dt.month == month,
            ds.isel(time=ds.time.dt.month == month).squeeze("time"),
            reindexed,
        )
    return reindexed


def assign_encoding(da, **kwargs):
    da = da.copy(deep=False)
    da.encoding.update(kwargs)
    return da


if __name__ == "__main__":
    ds = xr.open_zarr(SST_CLIMATOLOGY_ZARR)
    sst_knots = xr.cftime_range(
        "2016-12-01", periods=53, freq="MS", calendar="julian"
    ) + datetime.timedelta(days=14)
    reindexed_ssts = reindex_like_months(ds, sst_knots)
    interpolation_knots = {
        cftime.DatetimeJulian(2016, 12, 15): 0.0,
        cftime.DatetimeJulian(2017, 4, 1): 0.0,
        cftime.DatetimeJulian(2018, 4, 1): 0.5,
        cftime.DatetimeJulian(2019, 4, 1): 1.0,
        cftime.DatetimeJulian(2020, 4, 1): 1.5,
        cftime.DatetimeJulian(2021, 4, 1): 2.0,
        cftime.DatetimeJulian(2022, 4, 1): 2.5,
    }
    offsets = xr.DataArray(
        list(interpolation_knots.values()),
        dims=["time"],
        coords=[list(interpolation_knots.keys())],
    )
    offsets.interp(
        time=xr.cftime_range("2016-12-15", "2022-04-01", freq="D", calendar="julian")
    )

    spinup = xr.CFTimeIndex(
        [
            cftime.DatetimeJulian(2016, 12, 15),
            cftime.DatetimeJulian(2017, 1, 15),
            cftime.DatetimeJulian(2017, 2, 15),
            cftime.DatetimeJulian(2017, 3, 15),
            cftime.DatetimeJulian(2017, 4, 1),
        ]
    )
    ramp = xr.cftime_range(
        "2017-04-01", periods=49, freq="MS", calendar="julian"
    ) + datetime.timedelta(days=14)
    target_times = spinup.append(ramp)
    interpolated_offsets = offsets.interp(time=target_times)
    interpolated_ssts = reindexed_ssts.interp(time=target_times)

    pattern = interpolated_ssts + interpolated_offsets
    pattern = pattern.transpose("time", "latitude", "longitude")

    sst = assign_encoding(pattern.t, _FillValue=-99999.0).rename(
        "sea_surface_temperature"
    )
    sst["latitude"] = sst.latitude.assign_attrs(axis="Y")
    sst["longitude"] = sst.longitude.assign_attrs(axis="X")
    sst["time"] = assign_encoding(sst.time, dtype=float).assign_attrs(axis="T")
    sst_ds = sst.to_dataset()

    with dask.diagnostics.ProgressBar():
        sst_ds.to_netcdf("sst.nc", unlimited_dims=["time"])

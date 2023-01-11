# flake8: noqa
import xarray as xr
from datetime import datetime, timedelta
import vcm
import cftime
import numpy as np

# fine-res h500, PRESsfc, w500, TB, TMP500_300, and PWAT is at gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/
# fine-res PRATEsfc_coarse is at "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"

# coarse-res data is interpolated to a regular 3h grid, since runs did not evenly divide
# by 3h. Data for h500, PRESsfc, w500, TMPlowest, RH850, RH500, TMP500_300, and PWAT is available at e.g.
# gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/atmos_dt_atmos.zarr
# coarse-res PRATEsfc is available at gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/sfc_dt_atmos.zarr/
# similar coarse-res data is available in other subdirectories


def convert_fine(ds_atmos: xr.Dataset, ds_precip: xr.Dataset) -> xr.Dataset:
    # convert from 15-minute to 3-hourly
    assert (
        ds_atmos.time[0].values.item()
        == (ds_precip.time[0] + timedelta(minutes=45, hours=2)).values.item()
    )
    assert timedelta(
        seconds=int((ds_atmos.time[1] - ds_atmos.time[0]).values.item() / 1e9)
    ) == timedelta(hours=3)
    assert timedelta(
        seconds=int((ds_precip.time[1] - ds_precip.time[0]).values.item() / 1e9)
    ) == timedelta(minutes=15)
    assert np.sum(np.isnan(ds_precip["PRATEsfc_coarse"].values)) == 0
    ds_precip = (
        ds_precip.coarsen(time=3 * 4, boundary="trim")
        .mean()
        .rename({"grid_xt_coarse": "grid_xt", "grid_yt_coarse": "grid_yt"})
    )
    n_time = len(ds_atmos.time)
    return_ds = ds_atmos.assign(
        {
            "PRATEsfc": (
                ds_precip["PRATEsfc_coarse"].dims,
                ds_precip["PRATEsfc_coarse"].isel(time=slice(1, n_time + 1)).values,
            )
        }
    ).rename({"TB": "TMPlowest"})
    assert np.sum(np.isnan(return_ds["PRATEsfc"].values)) == 0
    return return_ds


def convert_coarse(ds_atmos: xr.Dataset, ds_precip: xr.Dataset) -> xr.Dataset:
    # coarse data has a different seasonal distribution than the fine data,
    # fine goes from august to august while coarse goes from august to january
    # so we chop off the extra seasonal portion for the coarse data
    return ds_atmos.assign({"PRATEsfc": ds_precip["PRATEsfc"]}).sel(
        time=slice(
            ds_atmos.time[0],
            cftime.DatetimeJulian(2023, 7, 31, 22, 0, 0, 0, has_year_zero=False),
        )
    )


def write(fine: xr.Dataset, coarse: xr.Dataset, name: str):
    date_string = datetime.now().strftime("%Y-%m-%d")
    fine_path = f"gs://vcm-ml-experiments/mcgibbon/{date_string}/fine-{name}.zarr"
    if not vcm.get_fs(fine_path).exists(fine_path):
        print(f"writing fine-res data to {fine_path}")
        fine.to_zarr(fine_path, consolidated=True)
    else:
        print(f"fine-res data already exists at {fine_path}, skipping")
    coarse_path = f"gs://vcm-ml-experiments/mcgibbon/{date_string}/coarse-{name}.zarr"
    if not vcm.get_fs(coarse_path).exists(coarse_path):
        print(f"writing coarse-res data to {coarse_path}")
        coarse.to_zarr(coarse_path, consolidated=True)
    else:
        print(f"coarse-res data already exists at {coarse_path}, skipping")


if __name__ == "__main__":
    ds_fine_atmos = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/"
    )
    ds_fine_precip = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"
    )
    ds_fine = convert_fine(ds_fine_atmos, ds_fine_precip)
    ds_fine.attrs[
        "source"
    ] = "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/"
    ds_fine.attrs[
        "source_PRATEsfc"
    ] = "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"
    ds_coarse_atmos = xr.open_zarr(
        "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/atmos_dt_atmos.zarr"
    )
    ds_coarse_precip = xr.open_zarr(
        "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/sfc_dt_atmos.zarr/"
    )
    ds_coarse = convert_coarse(ds_coarse_atmos, ds_coarse_precip)
    ds_coarse.attrs[
        "source"
    ] = "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/atmos_dt_atmos.zarr"
    ds_coarse.attrs[
        "source_PRATEsfc"
    ] = "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/sfc_dt_atmos.zarr/"
    write(ds_fine, ds_coarse, "0K")
    ds_fine_atmos = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/plus-8K/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/"
    )
    ds_fine_precip = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/plus-8K/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"
    )
    ds_fine = convert_fine(ds_fine_atmos, ds_fine_precip)
    ds_fine.attrs[
        "source"
    ] = "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/plus-8K/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr/"
    ds_fine.attrs[
        "source_PRATEsfc"
    ] = "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/plus-8K/C384-to-C48-diagnostics/gfsphysics_15min_coarse.zarr/"
    ds_coarse_atmos = xr.open_zarr(
        "gs://vcm-ml-experiments/spencerc/2021-08-11/n2f-25km-baseline-plus-8k/fv3gfs_run/atmos_dt_atmos.zarr"
    )
    ds_coarse_precip = xr.open_zarr(
        "gs://vcm-ml-experiments/spencerc/2021-08-11/n2f-25km-baseline-plus-8k/fv3gfs_run/sfc_dt_atmos.zarr/"
    )
    ds_coarse = convert_coarse(ds_coarse_atmos, ds_coarse_precip)
    ds_coarse.attrs[
        "source"
    ] = "gs://vcm-ml-experiments/spencerc/2021-08-11/n2f-25km-baseline-plus-8k/fv3gfs_run/atmos_dt_atmos.zarr"
    ds_coarse.attrs[
        "source_PRATEsfc"
    ] = "gs://vcm-ml-experiments/spencerc/2021-08-11/n2f-25km-baseline-plus-8k/fv3gfs_run/sfc_dt_atmos.zarr/"
    write(ds_fine, ds_coarse, "8K")

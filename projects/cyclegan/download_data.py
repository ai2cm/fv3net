import xarray as xr

# RENAMES = {
#     "TMP500_300": "tmp500_300",
#     "TB": "tb",
#     "PRESsfc": "pressfc",
# }

if __name__ == "__main__":
    # c48 = xr.open_zarr(
    #     "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/atmos_dt_atmos.zarr"  # noqa: E501
    # )
    c384 = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-01-04-1-year-C384-FV3GFS-simulations/unperturbed/C384-to-C48-diagnostics/atmos_8xdaily_coarse_interpolated.zarr"  # noqa: E501
    )  # .rename(RENAMES)
    # c48.to_zarr("c48_baseline.zarr")
    c384.to_zarr("c384_baseline.zarr")

import xarray as xr

c384 = xr.open_zarr(
    "gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_phys_3h_coarse.zarr"  # noqa: E501
).rename({"grid_xt_coarse": "x", "grid_yt_coarse": "y"})
c48 = xr.open_zarr(
    "gs://vcm-ml-intermediate/2021-10-12-PIRE-c48-post-spinup-verification/pire_atmos_phys_3h_coarse.zarr"  # noqa: E501
).rename({"grid_xt": "x", "grid_yt": "y"})

import xarray as xr
import numpy as np
import fsspec
from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from dask.diagnostics import ProgressBar


ENTRY = "10day_c48_PIRE_ccnorm_gfsphysics_15min_may2023"
VARIABLES = ["DSWRFsfc_coarse", "DLWRFsfc_coarse", "USWRFsfc_coarse", "PRATEsfc_coarse"]
RENAME = {
    "DSWRFsfc": "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",  # noqa: E501
    "DLWRFsfc": "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",  # noqa: E501
    "NSWRFsfc": "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
}
OUTPUT_PATH = "gs://vcm-ml-intermediate/2021-03-fine-res-surface-radiative-fluxes/10day-PIRE-ccnorm-coarsened-surface-radiative-fluxes-precip.zarr"  # noqa: E501

timestep_seconds = 900.0
m_per_mm = 1 / 1000


def add_total_precipitation(ds: xr.Dataset) -> xr.Dataset:
    total_precipitation = ds["PRATEsfc"] * m_per_mm * timestep_seconds
    total_precipitation = total_precipitation.assign_attrs(
        {"long_name": "precipitation increment to land surface", "units": "m"}
    )
    ds["total_precipitation"] = total_precipitation
    return ds.drop_vars("PRATEsfc")


def add_net_shortwave(ds: xr.Dataset) -> xr.Dataset:
    net_shortwave = ds["DSWRFsfc"] - ds["USWRFsfc"]
    net_shortwave = net_shortwave.assign_attrs(
        {
            "long_name": "net shortwave radiative flux at surface (downward)",
            "units": "W/m^2",
        }
    )
    ds["NSWRFsfc"] = net_shortwave
    return ds.drop_vars("USWRFsfc")


def cast_to_double(ds: xr.Dataset) -> xr.Dataset:
    new_ds = {}
    for name in ds.data_vars:
        if ds[name].values.dtype != np.float64:
            new_ds[name] = (
                ds[name]
                .astype(np.float64, casting="same_kind")
                .assign_attrs(ds[name].attrs)
            )
        else:
            new_ds[name] = ds[name]
    return xr.Dataset(new_ds).assign_attrs(ds.attrs)


if __name__ == "__main__":
    ds = (
        standardize_fv3_diagnostics(CATALOG[ENTRY].to_dask()[VARIABLES])
        .pipe(add_net_shortwave)
        .pipe(add_total_precipitation)
        .pipe(cast_to_double)
        .rename(RENAME)
    )
    with ProgressBar():
        mapper = fsspec.get_mapper(OUTPUT_PATH)
        ds.to_zarr(mapper, consolidated=True)

# Make a training dataset for a radiative flux prediction model.
import numpy as np
import xarray as xr
import intake
import fsspec
import click
import logging
from dask.distributed import Client
from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm.safe import get_variables

VERIFICATION_KEY = "40day_c48_gfsphysics_15min_may2020"
VERIFICATION_VARIABLES = ["DSWRFsfc", "DLWRFsfc", "DSWRFtoa"]
RENAME = {
    "DLWRFsfc": (
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
    )
}
STATE_VARIABLES = [
    "land_sea_mask",
    "surface_geopotential",
    "air_temperature",
    "specific_humidity",
]


def _verification_fluxes(dataset_key: str) -> xr.Dataset:
    try:
        ds = CATALOG[dataset_key].to_dask()
    except KeyError:
        ds = intake.open_zarr(dataset_key, consolidated=True).to_dask()
    ds = standardize_fv3_diagnostics(ds)
    ds = get_variables(ds, VERIFICATION_VARIABLES)
    ds = ds.assign(
        {
            "shortwave_transmissivity_of_atmospheric_column": _shortwave_transmissivity(
                ds["DSWRFsfc"], ds["DSWRFtoa"]
            )
        }
    ).drop_vars(["DSWRFsfc", "DSWRFtoa"])
    return _clear_encoding(ds.rename(RENAME))


def _shortwave_transmissivity(
    downward_shortwave_sfc: xr.DataArray, downward_shortwave_toa: xr.DataArray
) -> xr.DataArray:
    shortwave_transmissivity = downward_shortwave_sfc / downward_shortwave_toa
    shortwave_transmissivity = shortwave_transmissivity.where(
        downward_shortwave_toa > 0.0, 0.0
    )
    return shortwave_transmissivity.assign_attrs(
        {
            "long_name": "column shortwave transmissivity (sw_down_sfc / sw_down_toa)",
            "units": "-",
        }
    )


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    for var in ds.data_vars:
        ds[var].encoding = {}
    return ds


def _state(dataset_path: str, consolidated: bool) -> xr.Dataset:
    ds = intake.open_zarr(dataset_path, consolidated=consolidated).to_dask()
    ds = get_variables(ds, STATE_VARIABLES)
    return standardize_fv3_diagnostics(ds)


def cast_to_double(ds: xr.Dataset) -> xr.Dataset:
    new_ds = {}
    for name in ds.data_vars:
        if ds[name].dtype != np.float64:
            new_ds[name] = (
                ds[name]
                .astype(np.float64, casting="same_kind")
                .assign_attrs(ds[name].attrs)
            )
        else:
            new_ds[name] = ds[name]
    return xr.Dataset(new_ds).assign_attrs(ds.attrs)


@click.command()
@click.argument("state_path", type=str)
@click.argument("output_path", type=str)
@click.option(
    "--verification_key",
    type=str,
    default=VERIFICATION_KEY,
    help="Catalog key or path to radiative flux verification.",
)
def main(state_path, output_path, verification_key):

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training dataset generation.")

    client = Client()
    logging.info(f"See dask client dashboard: {client.dashboard_link}")

    verif_ds = _verification_fluxes(verification_key)

    logging.info(
        f"Writing training data zarr using state from {state_path} "
        f"and surface fluxes from {verification_key} "
        f"to output location {output_path}."
    )
    state_ds = _state(state_path, consolidated=True)
    state_chunks = state_ds.chunks
    ds = xr.merge([verif_ds, state_ds], join="inner")
    ds = ds.chunk(state_chunks)
    ds = cast_to_double(ds)

    mapper = fsspec.get_mapper(output_path)
    ds.to_zarr(mapper, consolidated=True)


if __name__ == "__main__":
    main()

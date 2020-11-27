import os
import subprocess
import tempfile
from typing import Sequence
import click
import numpy as np
import xarray as xr
from .gsutil import download_directory, upload_dir

MOSAIC_FILES = "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data"


def _standardize_dataset_for_fregrid(
    ds, x_dim, y_dim, x_interface_dim, y_interface_dim
):
    required_attrs = {
        x_dim: {"cartesian_axis": "X"},
        y_dim: {"cartesian_axis": "Y"},
        x_interface_dim: {"cartesian_axis": "X"},
        y_interface_dim: {"cartesian_axis": "Y"},
    }
    for dim, attrs in required_attrs.items():
        if dim in ds.dims:
            ds[dim] = ds[dim].assign_attrs(attrs)
            ds = ds.assign_coords({dim: np.arange(1.0, ds.sizes[dim] + 1)})
    if "time" in ds:
        ds["time"] = ds["time"].astype(float)
    return ds


def _write_dataset_to_tiles(ds, prefix):
    for tile in range(6):
        ds.isel(tile=tile).to_netcdf(f"{prefix}.tile{tile+1}.nc")


def _get_fregrid_args(
    input_mosaic, remap_file, input_file, output_file, scalar_fields, extra_args
):
    args = [
        "--input_mosaic",
        input_mosaic,
        "--remap_file",
        remap_file,
        "--input_file",
        input_file,
        "--output_file",
        output_file,
        "--scalar_field",
        ",".join(scalar_fields),
    ] + list(extra_args)
    return args


def fregrid(
    ds: xr.Dataset,
    x_dim: str = "x",
    y_dim: str = "y",
    x_interface_dim: str = "x_interface",
    y_interface_dim: str = "y_interface",
    scalar_fields: Sequence = None,
    extra_args=("--nlat", "180", "--nlon", "360"),
) -> xr.Dataset:
    """docstring"""
    resolution = f"C{ds.sizes[x_dim]}"
    mosaic_to_download = os.path.join(MOSAIC_FILES, resolution)

    if scalar_fields is None:
        scalar_fields = [
            v for v in ds.data_vars if {x_dim, y_dim, "tile"} <= set(ds[v].dims)
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mosaic = os.path.join(tmpdir, "mosaic")
        tmp_input = os.path.join(tmpdir, "input_data")
        tmp_output = os.path.join(tmpdir, "regridded_data.nc")

        fregrid_args = _get_fregrid_args(
            os.path.join(tmp_mosaic, "grid_spec.nc"),
            os.path.join(tmpdir, f"{resolution}_remap_file.nc"),
            tmp_input,
            tmp_output,
            scalar_fields,
            extra_args,
        )

        download_directory(mosaic_to_download, tmp_mosaic)
        horizontal_dims = [x_dim, y_dim, x_interface_dim, y_interface_dim]
        ds = _standardize_dataset_for_fregrid(ds, *horizontal_dims)
        _write_dataset_to_tiles(ds, tmp_input)
        subprocess.check_call(["fregrid"] + fregrid_args)
        ds_latlon = xr.open_dataset(tmp_output)
        return ds_latlon.rename({x_dim: "longitude", y_dim: "latitude"})


@click.command()
@click.argument("url")
@click.argument("output")
def fregrid_single_input(url: str, output: str):
    """Interpolate cubed sphere dataset at URL to lat-lon and save to OUTPUT"""
    with fsspec.open(url) as f:
        ds = xr.open_dataset(f)
        ds_latlon = fregrid(ds)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_latlon.to_netcdf(os.path.join(tmpdir, "data.nc"))
            upload_dir(os.path.join(tmpdir, "data.nc"), output)


if __name__ == "__main__":
    fregrid_single_input()

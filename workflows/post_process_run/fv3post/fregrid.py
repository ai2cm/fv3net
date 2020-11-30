import os
import subprocess
import tempfile
from typing import Sequence
import click
import numpy as np
import xarray as xr
from .gsutil import authenticate, download_directory, cp

MOSAIC_FILES = "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data"


def _standardize_dataset_for_fregrid(ds, x_dim, y_dim):
    required_attrs = {
        x_dim: {"cartesian_axis": "X"},
        y_dim: {"cartesian_axis": "Y"},
    }
    for dim, attrs in required_attrs.items():
        if dim in ds.dims:
            ds = ds.assign_coords({dim: np.arange(1.0, ds.sizes[dim] + 1)})
            ds[dim] = ds[dim].assign_attrs(attrs)
    if "time" in ds.coords:
        ds["time"].encoding["dtype"] = float
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
    scalar_fields: Sequence = None,
    extra_args=("--nlat", "180", "--nlon", "360"),
) -> xr.Dataset:
    """Interpolate dataset from cubed-sphere grid to lat-lon grid.
    
    Note:
        Saves dataset to disk and uses command-line fregrid tool to do interpolation.

    Args:
        ds: dataset to be interpolated. Must have 'tile' dimension.
        x_dim (optional): name of x-dimension. Defaults to 'x'.
        y_dim (optional): name of y-dimension. Defaults to 'y'.
        scalar_fields (optional): sequence of variable names to regrid. Defaults to all
            variables in ds whose dimensions include x_dim, y_dim and 'tile'.
        extra_args (optional): sequence of arguments to pass to command-line fregrid.
            Defaults to ("--nlat", "180", "--nlon", "360")."""
    authenticate()
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

        ds = _standardize_dataset_for_fregrid(ds, x_dim, y_dim)
        _write_dataset_to_tiles(ds, tmp_input)
        download_directory(mosaic_to_download, tmp_mosaic)
        subprocess.check_call(["fregrid"] + fregrid_args)
        ds_latlon = xr.open_dataset(tmp_output)
        return ds_latlon.rename({x_dim: "longitude", y_dim: "latitude"})


@click.command()
@click.argument("url")
@click.argument("output")
def fregrid_single_input(url: str, output: str):
    """Interpolate cubed sphere dataset at URL to lat-lon and save to OUTPUT"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp(url, os.path.join(tmpdir, "input.nc"))
        ds = xr.open_dataset(os.path.join(tmpdir, "input.nc"))
        ds_latlon = fregrid(ds)
        ds_latlon.to_netcdf(os.path.join(tmpdir, "data.nc"))
        cp(os.path.join(tmpdir, "data.nc"), output)


if __name__ == "__main__":
    fregrid_single_input()

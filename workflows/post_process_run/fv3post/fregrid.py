import os
import subprocess
import tempfile
from typing import Sequence
import click
import numpy as np
import xarray as xr
from .gsutil import authenticate, download_directory, cp


class Fregrid:
    mosaic_files_path = "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data"

    def __init__(self, resolution: str, nlat: int, nlon: int):
        """Cubed-sphere to lat-lon interpolation using the command-line fregrid tool.

        Note:
            Input mosaic file is downloaded from GCS and remapping coefficients are
            computed during object initialization.

        Args:
            resolution: one of "C48", "C96" or "C384".
            nlat: length of target latitude dimension.
            nlon: length of target longitude dimension.
        """
        authenticate()
        self.resolution = resolution
        self.nlat = nlat
        self.nlon = nlon
        mosaic_filenames = [f"{self.resolution}_grid.tile{n}.nc" for n in range(1, 7)]
        mosaic_filenames += ["grid_spec.nc"]

        # download mosaic and generate remapping file for future interpolation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_mosaic = os.path.join(tmpdir, "mosaic")
            tmp_remap = os.path.join(tmpdir, "remap.nc")
            download_directory(
                os.path.join(self.mosaic_files_path, resolution), tmp_mosaic
            )
            fregrid_args = self._get_fregrid_initialize_args(
                os.path.join(tmp_mosaic, "grid_spec.nc"), tmp_remap
            )
            subprocess.check_call(["fregrid"] + fregrid_args)
            self.mosaic = {
                filename: xr.open_dataset(os.path.join(tmp_mosaic, filename)).load()
                for filename in mosaic_filenames
            }
            self.remap = xr.open_dataset(tmp_remap).load()

    def interpolate(
        self,
        ds: xr.Dataset,
        x_dim: str = "x",
        y_dim: str = "y",
        scalar_fields: Sequence = None,
    ) -> xr.Dataset:
        """Interpolate dataset from cubed-sphere grid to lat-lon grid.

        Note:
            Saves dataset to disk and uses command-line fregrid to do interpolation.

        Args:
            ds: dataset to be interpolated. Must have 'tile' dimension.
            x_dim (optional): name of x-dimension. Defaults to 'x'.
            y_dim (optional): name of y-dimension. Defaults to 'y'.
            scalar_fields (optional): sequence of variable names to regrid. Defaults to
                all variables in ds whose dimensions include x_dim, y_dim and 'tile'."""
        if scalar_fields is None:
            scalar_fields = [
                v for v in ds.data_vars if {x_dim, y_dim, "tile"} <= set(ds[v].dims)
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_input = os.path.join(tmpdir, "input_data")
            tmp_regrid_file = os.path.join(tmpdir, "remap.nc")
            tmp_output = os.path.join(tmpdir, "regridded_data.nc")

            fregrid_args = self._get_fregrid_interpolate_args(
                os.path.join(tmpdir, "grid_spec.nc"),
                tmp_regrid_file,
                tmp_input,
                tmp_output,
                scalar_fields,
            )

            ds = self._standardize_dataset_for_fregrid(ds, x_dim, y_dim)
            for filename, mosaic_file in self.mosaic.items():
                mosaic_file.to_netcdf(os.path.join(tmpdir, filename))
            self.remap.to_netcdf(tmp_regrid_file)
            self._write_dataset_to_tiles(ds, tmp_input)
            subprocess.check_call(["fregrid"] + fregrid_args)
            ds_latlon = xr.open_dataset(tmp_output)
            return ds_latlon.rename(
                {
                    x_dim: "longitude",
                    y_dim: "latitude",
                    f"{x_dim}_bnds": "longitude_bnds",
                    f"{y_dim}_bnds": "latitude_bnds",
                }
            )

    @staticmethod
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

    @staticmethod
    def _write_dataset_to_tiles(ds, prefix):
        for tile in range(6):
            ds.isel(tile=tile).to_netcdf(f"{prefix}.tile{tile+1}.nc")

    def _get_fregrid_interpolate_args(
        self, mosaic_file, remap_file, input_file, output_file, scalar_fields
    ):
        args = [
            "--input_mosaic",
            mosaic_file,
            "--remap_file",
            remap_file,
            "--input_file",
            input_file,
            "--output_file",
            output_file,
            "--scalar_field",
            ",".join(scalar_fields),
            "--nlat",
            str(self.nlat),
            "--nlon",
            str(self.nlon),
        ]
        return args

    def _get_fregrid_initialize_args(self, mosaic_file, remap_file):
        args = [
            "--input_mosaic",
            mosaic_file,
            "--remap_file",
            remap_file,
            "--nlat",
            str(self.nlat),
            "--nlon",
            str(self.nlon),
        ]
        return args


@click.command()
@click.argument("url")
@click.argument("output")
def fregrid_single_input(url: str, output: str):
    """Interpolate cubed sphere dataset at URL to 1-degree lat-lon and save to OUTPUT.
    
    Assumes all tiles are contained in single netCDF file and input dimension names
    of "x", "y" and "tile"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp(url, os.path.join(tmpdir, "input.nc"))
        ds = xr.open_dataset(os.path.join(tmpdir, "input.nc"))
        resolution = f"C{ds.sizes['x']}"
        fregridder = Fregrid(resolution, 180, 360)
        ds_latlon = fregridder.interpolate(ds)
        ds_latlon.to_netcdf(os.path.join(tmpdir, "data.nc"))
        cp(os.path.join(tmpdir, "data.nc"), output)


if __name__ == "__main__":
    fregrid_single_input()

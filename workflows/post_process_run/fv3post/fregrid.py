import os
import subprocess
import tempfile
from typing import Sequence
import click
import numpy as np
import xarray as xr
from .gsutil import authenticate, download_directory, cp

MOSAIC_FILES_URL_DEFAULT = (
    "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data"
)


class FregridLatLon:
    def __init__(
        self,
        resolution: str,
        nlat: int,
        nlon: int,
        mosaic_files_url: str = MOSAIC_FILES_URL_DEFAULT,
    ):
        """Cubed-sphere to lat-lon interpolation using the command-line fregrid tool.

        Note:
            Input mosaic file is downloaded from GCS and remapping coefficients are
            computed during object initialization.

        Args:
            resolution: one of "C48", "C96" or "C384".
            nlat: length of target latitude dimension.
            nlon: length of target longitude dimension.
            mosaic_files_url: (optional) local or remote directory containing mosaic
                files. Defaults to 'gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-
                mosaic-data'.
        """
        self.resolution = resolution
        self.nlat = nlat
        self.nlon = nlon
        mosaic_files_url_for_resolution = os.path.join(mosaic_files_url, resolution)

        # download mosaic and generate remapping file for future interpolation
        with tempfile.TemporaryDirectory() as tmpdir:
            mosaic_dir = os.path.join(tmpdir, "mosaic")
            mosaic_grid_spec_path = os.path.join(mosaic_dir, "grid_spec.nc")
            remap_file_path = os.path.join(tmpdir, "remap.nc")

            download_directory(mosaic_files_url_for_resolution, mosaic_dir)
            args = self._get_initialize_args(mosaic_grid_spec_path, remap_file_path)
            subprocess.check_call(["fregrid"] + args)
            self.mosaic = {
                os.path.basename(path): xr.open_dataset(path).load()
                for path in self._get_mosaic_paths(mosaic_dir)
            }
            self.remap = xr.open_dataset(remap_file_path).load()

    def regrid(
        self,
        ds: xr.Dataset,
        x_dim: str = "x",
        y_dim: str = "y",
        scalar_fields: Sequence = None,
    ) -> xr.Dataset:
        """Regrid dataset from cubed-sphere grid to lat-lon grid.

        Note:
            Saves dataset to disk and uses command-line fregrid to do regridding.

        Args:
            ds: dataset to be regridded. Must have 'tile' dimension.
            x_dim (optional): name of x-dimension. Defaults to 'x'.
            y_dim (optional): name of y-dimension. Defaults to 'y'.
            scalar_fields (optional): sequence of variable names to regrid. Defaults to
                all variables in ds whose dimensions include x_dim, y_dim and 'tile'
                
        Returns:
            Dataset on a lat-lon grid. Horizontal dimension names are 'longitude' and
            'latitude'.
        """
        if scalar_fields is None:
            scalar_fields = [
                v for v in ds.data_vars if {x_dim, y_dim, "tile"} <= set(ds[v].dims)
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_prefix = os.path.join(tmpdir, "input_data")
            remap_file_path = os.path.join(tmpdir, "remap.nc")
            output_file_path = os.path.join(tmpdir, "regridded_data.nc")
            mosaic_grid_spec_path = os.path.join(tmpdir, "grid_spec.nc")

            args = self._get_regrid_args(
                mosaic_grid_spec_path,
                remap_file_path,
                input_prefix,
                output_file_path,
                scalar_fields,
            )

            ds = self._standardize_dataset_for_fregrid(ds, x_dim, y_dim)
            for filename, mosaic_file in self.mosaic.items():
                path = os.path.join(tmpdir, filename)
                mosaic_file.to_netcdf(path)
            self.remap.to_netcdf(remap_file_path)
            self._write_dataset_to_tiles(ds, input_prefix)
            subprocess.check_call(["fregrid"] + args)
            ds_latlon = xr.open_dataset(output_file_path)
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

    def _get_mosaic_paths(self, directory):
        mosaic_filenames = [f"{self.resolution}_grid.tile{n}.nc" for n in range(1, 7)]
        mosaic_filenames.append("grid_spec.nc")
        return [os.path.join(directory, filename) for filename in mosaic_filenames]

    def _get_regrid_args(
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

    def _get_initialize_args(self, mosaic_file, remap_file):
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
@click.argument("input_url")
@click.argument("output_url")
def fregrid_single_input(input_url: str, output_url: str):
    """Regrid cubed sphere dataset at INPUT_URL to 1-degree lat-lon and save to
    OUTPUT_URL.
    
    Assumes all tiles are contained in single netCDF file and regrids all variables
    whose dimensions include "x", "y" and "tile"."""
    authenticate()
    with tempfile.TemporaryDirectory() as tmpdir:
        cp(input_url, os.path.join(tmpdir, "input.nc"))
        ds = xr.open_dataset(os.path.join(tmpdir, "input.nc"))
        resolution = f"C{ds.sizes['x']}"
        fregridder = FregridLatLon(resolution, 180, 360)
        ds_latlon = fregridder.regrid(ds)
        ds_latlon.to_netcdf(os.path.join(tmpdir, "data.nc"))
        cp(os.path.join(tmpdir, "data.nc"), output_url)


if __name__ == "__main__":
    fregrid_single_input()

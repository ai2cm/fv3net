import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import cftime
import fsspec
import tempfile
import xarray as xr
import xarray_beam as xbeam
import argparse

"""
This script is tailored to provide ERA5 data regridded from 180x360 1deg regular
lat lon to C48 data for a
specific subset of variables used for creating an ocean RC model. The
chunking and machines used are tailored to the specific use case so adjustments
would be necessary for more variables or 3D data.

Hardcoded assumptions/directives:
- Input data is a Zarr store
- Input data is daily
- Input variables to regrid: SST, U10, V10, T2M (names under variable "use_vars")
- Output chunking assumes total number of days divisible by 64
"""

parser = argparse.ArgumentParser(description="Process daily averages")

parser.add_argument("input_path", help="Input Zarr path")
parser.add_argument("output_path", help="Output Zarr path")
parser.add_argument("--template_path", default=None, help="Path to daily template file")
parser.add_argument("--test", action="store_true", help="Run in test mode")


def preserve_attrs(ds, template):
    ds.attrs.update(template.attrs)
    for k, v in ds.items():
        v.attrs.update(template[k].attrs)
    return ds


def fregrid(key, ds):
    import xarray_beam as xbeam
    import subprocess
    import xarray as xr
    import netCDF4 as nc
    import cftime
    import numpy as np
    import logging
    import tempfile

    logger = logging.getLogger("fregrid_func")

    def save_netcdf_with_retry(ds, filename):
        num_tries = 5
        for i in range(num_tries):
            try:
                ds.to_netcdf(filename, engine="netcdf4")
                return
            except RuntimeError:
                subprocess.call(["rm", "-f", filename])
                logger.error(f"Retrying {filename} {i+1}/{num_tries}")
                pass

    def key_to_ncfile(key: xbeam.Key) -> str:
        out = ""
        for k, v in key.offsets.items():
            if not out:
                prefix = ""
            else:
                prefix = "_"
            out += f"{prefix}{k}{v:d}"
        return out + ".nc"

    def save_nc_int32_time(infile, outfile):
        in_nc = nc.Dataset(infile, "r")

        # load original time variable
        in_time = in_nc.variables["time"]
        in_time_values = in_time[:]
        as_dt = cftime.num2date(
            in_time_values, in_time.units, calendar=in_time.calendar
        )
        as_julian = cftime.date2num(as_dt, in_time.units, calendar="julian")
        in_nc.close()

        # save new file without time coordinate
        in_ds = xr.open_dataset(infile)
        save_netcdf_with_retry(in_ds.drop("time"), outfile)

        # add adjusted time dimension to the new file

        out_nc = nc.Dataset(outfile, "a")
        out_time = out_nc.createVariable("time", np.int32, ("time",))
        out_time[:] = as_julian
        for attr in in_time.ncattrs():
            if attr == "calendar":
                value = "julian"
            else:
                value = in_time.getncattr(attr)
            out_time.setncattr(attr, value)
        out_nc.close()
        return outfile

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = key_to_ncfile(key)
        out_file = f"{tmpdir}/{filename.replace('.nc', '_regrid.nc')}"
        file_path = f"{tmpdir}/{filename}"
        save_netcdf_with_retry(ds, file_path)
        to_process = save_nc_int32_time(
            file_path, f"{file_path.replace('.nc', '_int32.nc')}"
        )

        grid_file_path = "/work/source-grid-data/C48/grid_spec.nc"

        use_vars = [
            "sea_surface_temperature",
            "2m_temperature",
            "10m_v_component_of_wind",
            "10m_u_component_of_wind",
        ]

        subprocess.check_call(
            [
                "fregrid",
                "--input_mosaic",
                "/work/regular_lonlat_grid_mosaic.nc",
                "--output_mosaic",
                grid_file_path,
                "--input_file",
                f"{to_process}",
                "--output_file",
                f"{out_file}",
                "--scalar_field",
                ",".join(use_vars),
            ]
        )

        tiles = []
        for i in range(1, 7):
            fname = out_file.replace(".nc", f".tile{i}.nc")
            ds = xr.open_dataset(fname)
            tiles.append(ds.drop(["grid_xt", "grid_yt"]))
        tiles = xr.concat(tiles, dim="tile")
        tiles = tiles[use_vars].transpose("time", "tile", "grid_yt", "grid_xt").load()

    new_key = xbeam.Key(
        offsets={"time": key.offsets["time"], "grid_xt": 0, "grid_yt": 0, "tile": 0}
    )

    # convert numpy datetimes to cftime
    times = tiles.time.values
    times = [cftime.DatetimeJulian(x.year, x.month, x.day) for x in times]
    tiles = tiles.assign_coords({"time": times})
    return new_key, tiles


def main():
    args, beam_args = parser.parse_known_args()

    data = xr.open_zarr(args.input_path, consolidated=True)

    if args.test:
        data = data.isel(time=slice(0, 128))

    # build a template for the zarr output
    if args.template_path is not None:
        with tempfile.NamedTemporaryFile() as f:
            with fsspec.open(args.template_path) as f2:
                f.write(f2.read())
            f.flush()

            template = xr.open_dataset(f.name)
            template = xbeam.make_template(data)
            times = data.time
            # convert numpy datetimes to cftime
            as_dt = times.values.astype("M8[ms]").astype("O")
            times = [cftime.DatetimeJulian(x.year, x.month, x.day) for x in as_dt]
            template = template.expand_dims("tile", axis=1)
            template = template.isel(grid_xt=slice(0, 48), grid_yt=slice(0, 48))
            template = xr.concat([template] * 6, dim="tile")
            template = template.assign_coords({"time": times})
            template = preserve_attrs(template, data)
            template = template.chunk({"time": 64})
            print(template)
    else:
        template = None

    beam_options = PipelineOptions(beam_args)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | xbeam.DatasetToChunks(data)
            | xbeam.SplitChunks({"time": 4})
            | xbeam.ConsolidateChunks({"time": 64})
            | beam.MapTuple(fregrid)
            | xbeam.ChunksToZarr(args.output_path, template=template,)
        )


if __name__ == "__main__":
    main()

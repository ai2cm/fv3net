import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
import xarray as xr
import xarray_beam as xbeam
import argparse

"""
This script is tailored to provide daily averages from ERA5 data for a
specific subset of variables used for creating an ocean RC model. The
chunking and machines used are tailored to the specific use case so adjustments
would be necessary for more variables or 3D data.

Hardcoded assumptions/directives:
- Input data is a Zarr store
- Input data is hourly
- Input variables to regrid: SST, U10, V10, T2M
- Target time resolution: daily average
- Target grid: global regular 1deg lat/lon
- The loaded template file has same grid as target w/ same variables
- Nearest neighbor remapping
- Output chunking assumes total number of days divisible by 32
"""

parser = argparse.ArgumentParser(description="Process daily averages")

parser.add_argument("input_path", help="Input Zarr path")
parser.add_argument("--output_path", default="daily.zarr", help="Output Zarr path")
parser.add_argument(
    "--daily_template_path",
    default="single_day_template_lalo.nc",
    help="Path to daily template",
)
parser.add_argument("--test", action="store_true", help="Run in test mode")


def preserve_attrs(ds, template):
    ds.attrs.update(template.attrs)
    for k, v in ds.items():
        v.attrs.update(template[k].attrs)
    return ds


def cdo_regrid(key, ds):

    # Imports necessary for a Beam function since module scope not available in workers
    import xarray_beam as xbeam
    import subprocess
    import xarray as xr

    def key_to_ncfile(key: xbeam.Key) -> str:
        out = ""
        for k, v in key.offsets.items():
            if not out:
                prefix = ""
            else:
                prefix = "_"
            out += f"{prefix}{k}{v:d}"
        return out + ".nc"

    filename = key_to_ncfile(key)
    out_file = filename.replace(".nc", "_regrid.nc")
    ds.to_netcdf(filename)
    # Nearest neighbor remapping to global regular 1deg lat/lon grid
    subprocess.run(["cdo", "remapnn,global_1.0", filename, out_file])
    ds = xr.open_dataset(out_file)
    new_key = xbeam.Key(offsets={"time": key.offsets["time"], "lon": 0, "lat": 0})
    return new_key, ds


def main():
    args, beam_args = parser.parse_known_args()

    data = xr.open_zarr(args.input_path, chunks={"time": 24}, consolidated=True,)

    if args.test:
        output_num_days_chunk = 8
        data = data.isel(time=slice(0, 24 * output_num_days_chunk * 2))
    else:
        output_num_days_chunk = 32

    # Variables to regrid
    use_vars = ["sst", "u10", "v10", "t2m"]
    # variables that break the pipeline
    drop_vars = ["step", "depthBelowLandLayer", "entireAtmosphere", "number", "surface"]
    for_pred = data[use_vars].drop(drop_vars)

    # build a template for the zarr output
    single_day = xbeam.make_template(xr.open_dataset(args.daily_template_path))
    time_template = np.arange(
        data.time.isel(time=0).values,
        data.time.isel(time=-1).values,
        np.timedelta64(1, "D"),
    )
    num_times = len(time_template)
    template = xr.concat([single_day] * num_times, "time").assign_coords(
        {"time": time_template}
    )
    template = preserve_attrs(template, for_pred)
    template = xbeam.make_template(template)

    beam_options = PipelineOptions(beam_args)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | xbeam.DatasetToChunks(for_pred, chunks={"time": 24})
            | beam.MapTuple(
                lambda k, v: (
                    k.with_offsets(time=k.offsets["time"] // 24),
                    v.resample(time="1D").mean().compute(),
                )
            )
            | xbeam.ConsolidateChunks(target_chunks={"time": output_num_days_chunk})
            | beam.MapTuple(cdo_regrid)
            | xbeam.ChunksToZarr(
                args.output_path,
                template=template,
                zarr_chunks={"time": output_num_days_chunk},
            )
        )


if __name__ == "__main__":
    main()

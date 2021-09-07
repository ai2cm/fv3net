import logging
import os
import subprocess
import tempfile
import sys
import fsspec
import json
import numpy
import xarray
from fv3fit._shared.stacking import subsample
from loaders.batches import batches_from_mapper
from loaders.mappers import XarrayMapper
from joblib import Parallel, delayed

from fv3net.artifacts.resolve_url import resolve_url
from fv3net.artifacts.query import get_artifacts

# import apache_beam as beam

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def open_zarr(fs, url):
    return xarray.open_zarr(fs.get_mapper(url), consolidated=True)


def open_run(artifact):
    fs = fsspec.filesystem("gs")
    # avg = open_zarr(fs, os.path.join(url, "average.zarr"))
    return open_zarr(fs, os.path.join(artifact.path, "emulation.zarr"))


if __name__ == "__main__":
    random_state = numpy.random.RandomState(seed=0)
    subsample_size = 2560
    num_workers = 6
    variables = [
        "air_temperature",
        "canopy_water",
        "cloud_water_mixing_ratio",
        "eastward_wind",
        "land_sea_mask",
        "latent_heat_flux",
        "liquid_soil_moisture",
        "northward_wind",
        "ozone_mixing_ratio",
        "pressure_thickness_of_atmospheric_layer",
        "sea_ice_thickness",
        "sensible_heat_flux",
        "snow_depth_water_equivalent",
        "soil_temperature",
        "specific_humidity",
        "surface_pressure",
        "surface_temperature",
        "total_precipitation",
        "total_soil_moisture",
        "vertical_thickness_of_atmospheric_layer",
        "vertical_wind",
        "tendency_of_air_temperature_due_to_fv3_physics",
        "tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics",
        "tendency_of_eastward_wind_due_to_fv3_physics",
        "tendency_of_northward_wind_due_to_fv3_physics",
        "tendency_of_ozone_mixing_ratio_due_to_fv3_physics",
        "tendency_of_specific_humidity_due_to_fv3_physics",
    ]
    destination = resolve_url(
        "vcm-ml-archive", "online-emulator", tag="subsampled-data-v1"
    )

    matching_artifacts = [
        art
        for art in get_artifacts("gs://vcm-ml-experiments", ["online-emulator"])
        if art.tag.startswith("gfs-initialized-baseline")
    ]

    combined = xarray.concat([open_run(art) for art in matching_artifacts], dim="time")
    prefix = "emulator_"
    no_prefix = combined.rename(
        {v: v[len(prefix) :] for v in combined if v.startswith(prefix)}
    )
    data_mapping = XarrayMapper(no_prefix)
    batches = batches_from_mapper(data_mapping, variables)

    def stack(ds):
        return (
            ds.squeeze("time").stack(sample=["x", "y", "tile"]).transpose("sample", ...)
        )

    def _subsample(ds):
        return subsample(subsample_size, random_state, ds, dim="sample")

    def process(k, batches):
        logger.info(f"Processing {k}")
        ds = batches[k]
        out = _subsample(stack(ds))

        out.attrs["source_datasets"] = json.dumps(
            [str(art.path) for art in matching_artifacts]
        )
        out.attrs["git_version"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode()
        out.attrs["history"] = " ".join(sys.argv)
        out.attrs["working_directory"] = os.getcwd()

        fs = fsspec.filesystem("gs")

        with tempfile.NamedTemporaryFile() as f:
            out.reset_index("sample").to_netcdf(f.name)
            output_path = f"{destination}/window_{k:04d}.nc"
            fs.put(f.name, output_path)
            logger.info(f"{k}: done saving to {output_path}")

    # with beam.Pipeline() as p:
    #     indices = p | beam.Create(range(len(data_mapping))) | beam.Reshuffle()
    #     indices | "SaveData" >> beam.Map(process, batches=batches)

    Parallel(n_jobs=32, verbose=10)(
        delayed(process)(k, batches) for k in range(len(data_mapping))
    )

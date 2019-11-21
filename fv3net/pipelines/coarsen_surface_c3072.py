import apache_beam as beam
from vcm import cubedsphere
from vcm.cloud import gcs
from fv3net.pipelines import common
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


INPUT = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics"
OUTPUT = "gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened"
ORIGINAL_RESOLUTION = 3072
COARSENING = 8
PREFIX_DATA = "gfsphysics_15min_fine"
PREFIX_GRID_SPEC = "grid_spec"
AREA = "area"
DIM_KWARGS = {"x_dim": "grid_xt", "y_dim": "grid_yt"}
TILES = [1, 2, 3, 4, 5, 6]
SUBTILES = list(range(16))  # set to [None] if io_layout=1,1
output_resolution = int(ORIGINAL_RESOLUTION // COARSENING)
output_subdir = os.path.join(OUTPUT, f"C{output_resolution}")


def suffix(tile, subtile=None):
    if subtile is None:
        return f"tile{tile}.nc"
    else:
        return f"tile{tile}.nc.{subtile:04}"


def get_suffixes():
    return [suffix(t, st) for t in TILES for st in SUBTILES]


def filename(prefix, suffix):
    return f"{prefix}.{suffix}"


def url(bucket, prefix, suffix):
    return os.path.join(bucket, filename(prefix, suffix))


def download_to_file(url: str, dest: str):
    logging.info(f"Downloading from {url} to {dest}")
    blob = gcs.init_blob_from_gcs_url(url)
    blob.download_to_filename(dest)
    logging.info(f"Done downloading to {dest}")


def download_subtile(suffix: str):
    with tempfile.NamedTemporaryFile() as fdata, tempfile.NamedTemporaryFile() as fgrid:
        download_to_file(url(INPUT, PREFIX_DATA, suffix), fdata.name)
        download_to_file(url(INPUT, PREFIX_GRID_SPEC, suffix), fgrid.name)

        logger.info(f"{suffix}: opening xarray file")
        ds = xr.open_dataset(fdata.name)
        weights = xr.open_dataset(fgrid.name)[AREA]
        tile = suffix[4]

    for var in ds:
        yield {"tile": tile, "name": var}, (weights, ds[var].load())


def coarsen(key, val):
    weights, da = val
    ds_var_coarse = cubedsphere.weighted_block_average(
        da, weights, COARSENING, **DIM_KWARGS
    )
    return key, ds_var_coarse.to_dataset(name=da.name)


def _name(key):
    file = "{name}.tile{tile}.nc".format(**key)
    return os.path.join(output_subdir, file)


def run(beam_options):
    suffixes = get_suffixes()
    print(f"Processing {len(suffixes)} files")

    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(suffixes)
            | "Download Data" >> beam.ParDo(download_subtile)
            | "Coarsen" >> beam.MapTuple(coarsen)
            | common.CombineSubtilesByKey()
            | common.WriteToNetCDFs(_name)
        )


if __name__ == "__main__":
    """Main function"""
    logging.basicConfig(level=logging.DEBUG)
    beam_options = PipelineOptions(save_main_session=True)
    run(beam_options=beam_options)

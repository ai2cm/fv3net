import apache_beam as beam
from typing import Iterator
from vcm import cubedsphere
from vcm.cloud import gsutil, gcs
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)

INPUT = 'gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics'
OUTPUT = 'gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened'
ORIGINAL_RESOLUTION = 3072
COARSENING = 8
PREFIX_DATA = 'gfsphysics_15min_fine'
PREFIX_GRID_SPEC = 'grid_spec'
AREA = 'area'
DIM_KWARGS = {'x_dim': 'grid_xt', 'y_dim': 'grid_yt'}
TILES = [1, 2, 3, 4, 5, 6]
SUBTILES = list(range(16))  # set to [None] if io_layout=1,1
output_resolution = int(ORIGINAL_RESOLUTION // COARSENING)
output_subdir = os.path.join(OUTPUT, f'C{output_resolution}')


def suffix(tile, subtile=None):
    if subtile is None:
        return f'tile{tile}.nc'
    else:
        return f'tile{tile}.nc.{subtile:04}'


def get_suffixes():
    return [suffix(t, st) for t in TILES for st in SUBTILES]


def filename(prefix, suffix):
    return f'{prefix}.{suffix}'


def url(bucket, prefix, suffix):
    return os.path.join(bucket, filename(prefix, suffix))


def coarsen_file(suffix) -> Iterator[xr.Dataset]:
    with tempfile.NamedTemporaryFile() as fdata, tempfile.NamedTemporaryFile() as fgrid:
        gsutil.copy(url(INPUT, PREFIX_DATA, suffix), fdata.name, check_hashes=False)
        gsutil.copy(url(INPUT, PREFIX_GRID_SPEC, suffix), fgrid.name, check_hashes=False)

        ds = xr.open_dataset(fdata.name)
        weights = xr.open_dataset(fgrid.name)[AREA]
        for var in ds:
            coarse_var_filename = filename(f'{PREFIX_DATA}_{var}', suffix)
            remote_path = os.path.join(output_subdir, var, coarse_var_filename)
            ds_var_coarse = cubedsphere.weighted_block_average(ds[[var]],
                                                               weights,
                                                               COARSENING,
                                                               **DIM_KWARGS)
            ds_var_coarse.attrs = ds[var].attrs
            ds_var_coarse.to_netcdf(coarse_var_filename)
            gcs.copy(coarse_var_filename, remote_path)
            os.remove(coarse_var_filename)

    yield


def run(beam_options):
    suffixes = get_suffixes()
    print(f"Processing {len(suffixes)} files")
    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create(suffixes)
         | 'CoarsenFile' >> beam.ParDo(coarsen_file)
         )


if __name__ == '__main__':
    """Main function"""
    beam_options = PipelineOptions(save_main_session=True)
    run(beam_options=beam_options)

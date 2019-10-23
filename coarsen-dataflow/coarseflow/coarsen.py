import tempfile
import tarfile
import logging
import subprocess
from urllib import parse
from pathlib import Path
from collections import defaultdict
from itertools import product
from typing import Iterable

import apache_beam
from apache_beam.pvalue import PCollection  # type: ignore

from google.cloud.storage import Client, Bucket, Blob

from src.fv3 import coarsen as fv3net_coarsen
from src.data import cubedsphere
from src.fv3.docker import save_tiles_separately

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addFilter(logging.Filter(__name__))

"""
Full scaling would involve sending the netCDF files workers individually to coarsen
and then regathering files to combine in another stage of the pipeline.  Then output
them.

Simple way is since we are downloading to local instance.  Just do all the coarsening
and combining there.  Then reupload to google cloud.
"""

# TODO: Figure out how to specify tile inputs? Hard-coded currently

@apache_beam.typehints.with_input_types(str)
@apache_beam.typehints.with_output_types(None)
class CoarsenTimestep(apache_beam.DoFn):
    def __init__(self):
        super().__init__()

    def process(self, element):

        logger.info(f'Processing timestep tar: {element}')

        # TODO: Just pass the blob as the element, unless it's not picklable?
        parsed_gs_path = parse.urlsplit(element)
        bucket_name = parsed_gs_path.netloc
        blob_name = parsed_gs_path.path.lstrip('/')

        source_blob = self._init_blob(bucket_name, blob_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            
            logger.debug(f'Using temporary directory {tmpdir}')
            tarball_path = self._download_tar(source_blob, tmpdir, blob_name)
            extracted_path = self._extract_timestep_tar(tarball_path)

            # Temp for runthrough
            # extracted_path = Path(tmpdir, '20160801.003000')
            # import os
            # os.symlink('/home/andrep/repos/fv3net/data/tarball/20160801.003000', 
            #            extracted_path.as_posix())
        

            # Download all the gridspecs for C3072
            # TODO: Hardcoded for now, gridspecs should have res-specific dirs
            blob_dir = '2019-10-03-X-SHiELD-C3072-to-C384-diagnostics'
            ntiles = 6
            nsubtiles = 16
            tiles = range(1, ntiles+1)
            subtiles = range(nsubtiles)

            spec_paths = self._download_gridspecs(bucket_name, blob_dir,
                                                  tiles, subtiles, Path(tmpdir))
            
            
            coarsen_args = self._gen_coarsen_args(tiles, subtiles, extracted_path, spec_paths)
            # do the coarsen_sfc
            combined_dataset = fv3net_coarsen.coarsen_sfc_data_in_directory(coarsen_args, 
                                                                            method='median', 
                                                                            factor=8)

            # # Create output directory
            coarse_out_dir = Path(tmpdir).joinpath('coarse_output').joinpath(extracted_path.name)
            coarse_out_dir.mkdir(parents=True)
            
            # Save coarsened surface data out to directory as tiled files
            # TODO: include destination res in filename?
            save_tiles_separately(combined_dataset, 
                                  f'{extracted_path.name}.sfc_data_coarse.res', 
                                  coarse_out_dir.as_posix())

            # TODO: tmp for testing
            # coarse_out_dir = Path('/home/andrep/repos/fv3net/data/coarse_output/20160801.003000')

            # TODO: For now data has already been coarsened to 384, save all other variables
            # components = ['fv_core', 'fv_srf_wnd', 'fv_tracer']
            # for component in components:
            #     curr_prefix_fname = f'{extracted_path.name}.{component}_coarse.res'
            #     curr_prefix = extracted_path.joinpath(curr_prefix_fname)
            #     ds = cubedsphere.open_cubed_sphere(curr_prefix.as_posix())
            #     save_tiles_separately(ds, curr_prefix_fname, coarse_out_dir.as_posix())

            # tar coarsened data and put file on cloud storage
            coarse_out_tar_path = self._tar_coarsened_timestep(coarse_out_dir)

            dest_blob_name = ('2019-10-19-X-SHiELD-C3072-to-C384-diagnostics-wCoarseSfc'
                              f'/{coarse_out_tar_path.name}')
            dest_blob = self._init_blob(bucket_name, dest_blob_name)
            dest_blob.upload_from_filename(coarse_out_tar_path.as_posix(), content_type='application/tar')

    def _init_blob(self, bucket_name: str, blob_name: str) -> Blob:
        logger.debug(f'Initializing GCS Blob.  bucket={bucket_name}, blob={blob_name}')
        bucket = Bucket(Client(), bucket_name)
        return Blob(blob_name, bucket)

    
    def _download_gridspecs(self, bucket_name: str, blob_dir: str,
                            tiles: Iterable[int], subtiles: Iterable[int], 
                            output_dir: Path) -> defaultdict:

        logger.info('Downloading grid_spec files for coarsening.')
        
        grid_spec_fname = 'grid_spec.tile{:d}.nc.{:04d}'

        spec_paths_by_tile = defaultdict(dict)
        for tile in tiles:
            for subtile in subtiles:
                curr_spec_name = grid_spec_fname.format(tile, subtile)
                curr_blob_name = blob_dir + '/' + curr_spec_name
                spec_out_path = output_dir.joinpath(curr_spec_name)
                logger.debug(f'Downloading specfile: {spec_out_path}')
                spec_paths_by_tile[tile][subtile] = spec_out_path.as_posix()

                grid_spec_blob = self._init_blob(bucket_name, curr_blob_name)
                grid_spec_blob.download_to_filename(spec_out_path)

        return spec_paths_by_tile


    def _gen_coarsen_args(self, tiles: Iterable[int], subtiles: Iterable[int], 
                          extracted_dir: Path, spec_paths: dict):
        
        # TODO: this is a temporary for the partially regridded sfc data?
        extracted_dir.name
        logger.debug(f'Generating coarsen arguments for sfc data:\n'
                     f'\ttiles = {tiles}\n',
                     f'\tsubtiles = {subtiles}\n',
                     f'\tout_dir = {extracted_dir}')

        args = []
        for tile, subtile in product(tiles, subtiles):
            sfc_data_fname = f'{extracted_dir.name}.sfc_data.tile{tile}.nc.{subtile:04d}'
            sfc_data_path = Path(extracted_dir, sfc_data_fname).as_posix()
            args.append((tile, subtile, sfc_data_path, spec_paths[tile][subtile]))

        return args

    @staticmethod
    def _download_tar(source_blob: Blob, out_dir: str, filename: str) -> Path:
        logger.info(f'Downloading tar ({filename}) from remote storage.')

        out_dir = Path(out_dir)
        filename = Path(filename)
        download_path = out_dir.joinpath(filename)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Tarfile download path: {download_path}')

        source_blob.chunk_size = 128 * 2**20 # 128 MB chunks
        with open(download_path.as_posix(), mode='wb') as f:
            source_blob.download_to_file(f)
        return download_path

    @staticmethod
    def _extract_timestep_tar(downloaded_tar_path: Path) -> Path:

        logger.info('Extracting tar file...')

        
        # with suffix [blank] removes file_ext and uses filename as untar dir
        extract_dir = downloaded_tar_path.with_suffix('')
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Destination directory for tar extraction: {extract_dir}')
        subprocess.call(['tar', '-xf', downloaded_tar_path.as_posix(), 
                         '-C', extract_dir.as_posix()])

        return extract_dir

    @staticmethod
    def _tar_coarsened_timestep(dir_to_tar: Path) -> Path:
        logger.info('Re-tarring coarsened data...')
        tar_out = dir_to_tar.with_suffix(dir_to_tar.suffix + '.tar')
        logger.debug(f'Coarsened data tarfile: {tar_out}')
        subprocess.call(['tar', '-cf', tar_out.as_posix(), dir_to_tar.as_posix()])
        

        return tar_out


# if __name__ == '__main__':

#     coarse_op = CoarsenTimestep()
#     coarse_op.process('gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/20160801.003000.tar')
    
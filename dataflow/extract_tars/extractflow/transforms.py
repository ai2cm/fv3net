import tempfile
import logging
import subprocess
import shutil
import logging
import apache_beam
from pathlib import Path
from itertools import product
from apache_beam.pvalue import PCollection  # type: ignore
from apache_beam.utils import retry

import dataflow_utils.utils as utils
import dataflow_utils.gcs_utils as gcs_utils
from dataflow_utils.gcs_utils import GCSLister

from google.cloud.storage import Client, Bucket, Blob

delay_kwargs = dict(initial_delay_secs=30, num_retries=8)
logger = logging.getLogger(__name__)

# wrap sensitive functions with retry decorator
download_blob_to_file = retry.with_exponential_backoff(**delay_kwargs)(gcs_utils.download_blob_to_file)
upload_dir_to_gcs = retry.with_exponential_backoff(**delay_kwargs)(gcs_utils.upload_dir_to_gcs)

@apache_beam.typehints.with_input_types(str)
@apache_beam.typehints.with_output_types(None)
class ExtractAndUploadTimestepWithC3072SurfaceData(apache_beam.DoFn):
    def __init__(self, output_prefix: str):
        super().__init__()
        self.output_prefix = output_prefix

    def process(self, element):
        
        with tempfile.TemporaryDirectory() as tmpdir:

            timestep_blob = gcs_utils.init_blob_from_gcs_url(element)
            filename = Path(timestep_blob.name).name
            downloaded_timestep = download_blob_to_file(timestep_blob, tmpdir, filename)
            untarred_timestep = utils.extract_tarball_to_path(downloaded_timestep)

            current_timestep = untarred_timestep.name
            c384_path = Path(tmpdir, 'coarsened', 'C384', current_timestep)
            c384_path.mkdir(parents=True, exist_ok=True)
            c3702_path = Path(tmpdir, 'coarsened', 'C3702', current_timestep)
            c3702_path.mkdir(parents=True, exist_ok=True)

            # move highres sfc data to separate dir
            for sfc_file in untarred_timestep.glob('*.sfc_data.*'):

                destination = c3702_path.joinpath(sfc_file.name)
                shutil.move(sfc_file, destination)

            # upload highres sfc data
            c3702_blob_prefix = c3702_path.relative_to(c3702_path.parent.parent)
            c3702_blob_prefix = str(Path(self.output_prefix, c3702_blob_prefix))
            upload_dir_to_gcs('vcm-ml-data', c3702_blob_prefix, c3702_path)

            # upload pre-coarsened files to timestep
            c384_blob_prefix = c384_path.relative_to(c384_path.parent.parent)
            c384_blob_prefix = str(Path(self.output_prefix, c384_blob_prefix))
            upload_dir_to_gcs('vcm-ml-data', c384_blob_prefix, untarred_timestep)

            logging.info(f'Upload of untarred timestep successful ({current_timestep})')

class InvalidArgumentError(retry.PermanentException, ValueError):
    """Error Class that bypasses the retry operation"""
    pass

@retry.with_exponential_backoff(**delay_kwargs)
def not_finished_with_tar_extract(timestep_gcs_url: str, output_prefix: str,
                                  num_tiles: int = 6, num_subtiles: int = 16):
    """
    This function is currently particular to checking output from tarballs with
    high-res surface data and c384 everything else.

    Don't use for general checks of successful extraction.
    """
    # TODO: Should probably make sure test includes missing cases for all domains
    logger.info(f'Checking for successful extraction of {timestep_gcs_url}')
    bucket_name, blob_name = gcs_utils.parse_gcs_url(timestep_gcs_url)
    timestep = Path(blob_name).with_suffix('').name
    output_c3702_blob_prefix = Path(output_prefix, 'C3702', timestep)
    output_c384_blob_prefix = Path(output_prefix, 'C384', timestep)
    filename_template = f'{timestep}.{{data_domain}}.tile{{tile:d}}.nc.{{subtile:04d}}'

    def _check_for_all_tiles(data_domain: str, output_prefix: Path):
        # If number of tiles is less than 1 there's nothing to check
        if num_tiles < 1 or num_subtiles < 1:
            raise InvalidArgumentError(
                'Tile check requires a positive number of tiles and'
                ' subtiles to perform file existence checks.'
            )

        lister = GCSLister(Client(), bucket_name)
        existing_blob_names = [gcs_utils.parse_gcs_url(gcs_url)[1]  # 2nd element is blob name
                               for gcs_url in lister.list(prefix=output_prefix)]

        tiles = range(1, num_tiles + 1)
        subtiles = range(num_subtiles)
        all_exist = True
        for tile, subtile in product(tiles, subtiles):
            filename = filename_template.format(data_domain=data_domain,
                                                tile=tile,
                                                subtile=subtile)
            to_check_blob_name = str(output_prefix.joinpath(filename))
            does_blob_exist = to_check_blob_name in existing_blob_names
            all_exist &= does_blob_exist
            
            if not does_blob_exist:
                logger.debug(f'Missing blob detected in timestep {timestep}: '
                             f'{to_check_blob_name}')
            
        return all_exist
    
    sfc_files_ok = _check_for_all_tiles('sfc_data', output_c3702_blob_prefix)

    domain_list = ['fv_core_coarse.res', 'fv_srf_wnd_coarse.res',
                   'fv_tracer_coarse.res']
    domain_ok = {}
    for domain in domain_list:
        domain_ok[domain] = _check_for_all_tiles(domain, output_c384_blob_prefix)

    coupler_filename = f'{timestep}.coupler.res'
    coupler_blob_name = output_c384_blob_prefix.joinpath(coupler_filename)
    coupler_blob = gcs_utils.init_blob(bucket_name, str(coupler_blob_name))
    coupler_ok = coupler_blob.exists()

    domain_ok['sfc_data'] = sfc_files_ok
    domain_ok['coupler'] = coupler_ok

    files_ok = True
    for domain, ok in domain_ok.items():
        logger.debug(f'Extraction status successful? {domain}={ok}')
        files_ok &= ok

    # Filter transform removes false values. Pass thru if files are not ok
    do_extract = not files_ok
    logger.info(f'Continue extracting timestep ({timestep})? {do_extract}')

    return do_extract


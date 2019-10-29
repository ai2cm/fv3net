import tempfile
import logging
import subprocess
import shutil
import logging
from pathlib import Path
import apache_beam
from apache_beam.pvalue import PCollection  # type: ignore

import coarseflow.utils as cfutils
from coarseflow.file_lister import GCSLister

from google.cloud.storage import Client, Bucket, Blob

logger = logging.getLogger(__name__)


@apache_beam.typehints.with_input_types(str)
@apache_beam.typehints.with_output_types(None)
class ExtractAndUploadTimestepWithC3072SurfaceData(apache_beam.DoFn):
    def __init__(self, output_prefix: str):
        super().__init__()
        self.output_prefix = output_prefix

    def process(self, element):
        
        with tempfile.TemporaryDirectory() as tmpdir:

            timestep_blob = cfutils.init_blob_from_gcs_url(element)
            filename = Path(timestep_blob.name).name
            downloaded_timestep = cfutils.download_blob_to_file(timestep_blob, tmpdir, filename)
            untarred_timestep = cfutils.extract_tarball_to_path(downloaded_timestep)

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
            cfutils.upload_dir_to_gcs('vcm-ml-data', c3702_blob_prefix, c3702_path)

            # upload pre-coarsened files to timestep
            c384_blob_prefix = c384_path.relative_to(c384_path.parent.parent)
            c384_blob_prefix = str(Path(self.output_prefix, c384_blob_prefix))
            cfutils.upload_dir_to_gcs('vcm-ml-data', c384_blob_prefix, untarred_timestep)


def not_finished_with_tar_extract(timestep_gcs_url: str, output_prefix: str,
                                  num_tiles: int = 6, num_subtiles: int = 16):
    """
    This function is currently particular to checking output from tarballs with
    high-res surface data and c384 everything else.

    Don't use for general checks of successful extraction.
    """
    # TODO: Should probably make sure test includes missing cases for all domains
    logger.info(f'Checking for successful extraction of {timestep_gcs_url}')
    bucket_name, blob_name = cfutils.parse_gcs_url(timestep_gcs_url)
    timestep = Path(blob_name).with_suffix('').name
    output_c3702_blob_prefix = Path(output_prefix, 'C3702', timestep)
    output_c384_blob_prefix = Path(output_prefix, 'C384', timestep)
    filename_template = f'{timestep}.{{data_domain}}.tile{{tile:d}}.nc.{{subtile:04d}}'

    def _check_for_all_tiles(data_domain: str, output_prefix: Path):
        # If number of tiles is less than 1 there's nothing to check
        if num_tiles < 1 or num_subtiles < 1:
            all_exist = False
        else:
            all_exist = True

        for tile in range(1, num_tiles + 1):
            for subtile in range(num_subtiles):
                filename = filename_template.format(data_domain=data_domain,
                                                    tile=tile,
                                                    subtile=subtile)
                to_check_blob_name = str(output_prefix.joinpath(filename))
                blob = cfutils.init_blob(bucket_name, to_check_blob_name)
                all_exist &= blob.exists()

        return all_exist
    
    sfc_files_ok = _check_for_all_tiles('sfc_data', output_c3702_blob_prefix)

    domain_list = ['fv_core_coarse.res', 'fv_srf_wnd_coarse.res',
                   'fv_tracer_coarse.res']
    domain_ok = {}
    for domain in domain_list:
        domain_ok[domain] = _check_for_all_tiles(domain, output_c384_blob_prefix)

    coupler_filename = f'{timestep}.coupler.res'
    coupler_blob_name = output_c384_blob_prefix.joinpath(coupler_filename)
    coupler_blob = cfutils.init_blob(bucket_name, str(coupler_blob_name))
    coupler_ok = coupler_blob.exists()

    domain_ok['sfc_data'] = sfc_files_ok
    domain_ok['coupler'] = coupler_ok

    files_ok = True
    for domain, ok in domain_ok.items():
        logger.debug(f'Extraction status successful? {domain}={ok}')
        files_ok &= ok

    # Filter transform removes false values. Pass thru if files are not ok
    do_extract = not files_ok
    logger.debug(f'Continue extracting timestep? {do_extract}')

    return do_extract


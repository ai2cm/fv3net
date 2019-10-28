from urllib import parse
from typing import Tuple, Optional, List
from pathlib import Path
import dask.bag as db
import subprocess
import logging

from google.cloud.storage import Client, Bucket, Blob

logger = logging.getLogger(__name__)


def init_blob(bucket_name: str, blob_name: str) -> Blob:
    logger.debug(f'Initializing GCS Blob.  bucket={bucket_name}, blob={blob_name}')
    bucket = Bucket(Client(), bucket_name)
    return Blob(blob_name, bucket)


def parse_gcs_url(gcs_url: str) -> Tuple[str]:
    parsed_gs_path = parse.urlsplit(gcs_url)
    bucket_name = parsed_gs_path.netloc
    blob_name = parsed_gs_path.path.lstrip('/')

    return bucket_name, blob_name


def init_blob_from_gcs_url(gcs_url: str) -> Blob:

    bucket_name, blob_name = parse_gcs_url(gcs_url)
    return init_blob(bucket_name, blob_name)


def download_blob_to_file(source_blob: Blob, out_dir: str, filename: str) -> Path:
    logger.info(f'Downloading tar ({filename}) from remote storage.')

    out_dir = Path(out_dir)
    filename = Path(filename)
    download_path = out_dir.joinpath(filename)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f'Tarfile download path: {download_path}')

    source_blob.chunk_size = 128 * 2**20 # 128 MB chunks
    with open(download_path, mode='wb') as f:
        source_blob.download_to_file(f)
    return download_path


def extract_tarball_to_path(
    source_tar_path: Path,
    extract_to_dir: Optional[Path] = None,
    ) -> Path:

    logger.info('Extracting tar file...')

    # with suffix [blank] removes file_ext and uses filename as untar dir
    src_tar_no_file_ext = source_tar_path.with_suffix('')
    if extract_to_dir is None:
        # Untar creates folder with tar filename by default so use parent
        extract_to_dir = source_tar_path.parent
    
    extract_to_dir.mkdir(parents=True, exist_ok=True)
    destination_tar_dir = extract_to_dir.joinpath(src_tar_no_file_ext.name)

    logger.debug(f'Destination directory for tar extraction: {extract_to_dir}')
    tar_commands = ['tar', '-xf', source_tar_path, '-C', extract_to_dir]
    subprocess.call(tar_commands)

    return destination_tar_dir


def upload_dir_to_gcs(bucket_name: str, blob_prefix: str, source_dir: Path) -> None:
    """
    Uploads all files in specified directory to GCS directory
    """

    upload_args = [(bucket_name, blob_prefix, filepath)
                   for filepath in source_dir.glob('*')
                   if filepath.is_file()]
    upload_args_bag = db.from_sequence(upload_args)
    upload_args_bag.map(_upload_process).compute()

def _upload_process(args):
    bucket_name, blob_prefix, filepath = args

    filename = filepath.name
    blob_name = blob_prefix + '/' + filename
    destination_blob = init_blob(bucket_name, blob_name)
    destination_blob.upload_from_filename(str(filepath))




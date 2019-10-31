from urllib import parse
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Iterable
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


def upload_dir_to_gcs(bucket_name: str, blob_prefix: str, source_dir: Path) -> None:
    """
    Uploads all files in specified directory to GCS directory
    """
    logger.info(f'Uploading timestep to gcs (blob_prefix={blob_prefix})')
    logger.debug(f'GCS bucket = {bucket_name}')
    logger.debug(f'Source local dir = {source_dir}')
    upload_args = [(bucket_name, blob_prefix, filepath)
                   for filepath in source_dir.glob('*')
                   if filepath.is_file()]
    upload_args_bag = db.from_sequence(upload_args)
    upload_args_bag.map(_upload_process).compute(scheduler='single-threaded')


def _upload_process(args):
    bucket_name, blob_prefix, filepath = args

    filename = filepath.name
    blob_name = blob_prefix + '/' + filename
    destination_blob = init_blob(bucket_name, blob_name)
    destination_blob.upload_from_filename(str(filepath))


class FileLister(ABC):
    @abstractmethod
    def list(self, prefix=None, file_extension=None) -> Iterable[str]:
        pass


class GCSLister(FileLister):
    def __init__(self, client: Client, bucket: str):
        self.client = client
        self.bucket = bucket

    def list(self, 
        prefix: str = None, 
        file_extension: str = None
    ) -> Iterable[str]:
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        for blob in blobs:
            
            # filter specific extensions
            if file_extension is not None:
                blob_ext_name = blob.name.split('.')[-1]
                if file_extension.strip('.') != blob_ext_name:
                    continue
            
            yield f"gs://{blob.bucket.name}/{blob.name}"

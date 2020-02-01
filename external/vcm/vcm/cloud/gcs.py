import logging
import os
from pathlib import Path
from typing import Iterable, Tuple
from urllib import parse

import dask.bag as db
from google.cloud.storage import Blob, Bucket, Client

logger = logging.getLogger(__name__)


def init_blob(bucket_name: str, blob_name: str) -> Blob:
    logger.debug(f"Initializing GCS Blob.  bucket={bucket_name}, blob={blob_name}")
    bucket = Bucket(Client(), bucket_name)
    return Blob(blob_name, bucket)


def parse_gcs_url(gcs_url: str) -> Tuple[str]:
    parsed_gs_path = parse.urlsplit(gcs_url)
    bucket_name = parsed_gs_path.netloc
    blob_name = parsed_gs_path.path.lstrip("/")

    return bucket_name, blob_name


def init_blob_from_gcs_url(gcs_url: str) -> Blob:

    bucket_name, blob_name = parse_gcs_url(gcs_url)
    return init_blob(bucket_name, blob_name)


def download_blob_to_file(source_blob: Blob, out_dir: str, filename: str) -> Path:
    logger.info(f"Downloading ({filename}) from remote storage.")

    out_dir = Path(out_dir)
    filename = Path(filename)
    download_path = out_dir.joinpath(filename)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"File download path: {download_path}")

    source_blob.chunk_size = 128 * 2 ** 20  # 128 MB chunks
    with open(download_path, mode="wb") as f:
        source_blob.download_to_file(f)
    return download_path


def download_all_bucket_files(
    gcs_url: str, out_dir_prefix: str, include_parent_in_stem=True
):
    """
    Download all the GCS files and directories within a given GCS directory

    Args:
        gcs_url: Full path to the GCS directory containing contents to download
        out_dir_prefix: Local path to place downloaded directory contents
        include_parent_in_stem: Include the specified target directory whose contents
            are being downloaded in local output.
            
            E.g., for
            gcs_url = gs://bucket/download_my/contents/
            output_dir_prefix = /tmp
            
            including the parent would result in the directory structure of
            /tmp/contents/...
    """
    bucket_name, blob_prefix = parse_gcs_url(gcs_url)
    blob_gcs_paths = list_bucket_files(Client(), bucket_name, prefix=blob_prefix, is_dir=True)
    blob_gcs_paths = list(blob_gcs_paths)
    
    if not blob_gcs_paths:
        raise ValueError(f"No files found under directory gs://{bucket_name}/{blob_prefix}")
    parent_dirname = Path(blob_prefix).name
    logger.debug(f"Downloading files from bucket prefix: {blob_prefix}")

    for blob_url in blob_gcs_paths:
        _, blob_path = parse_gcs_url(blob_url)
        filename = Path(blob_path).name
        full_dir = str(Path(blob_path).parent)
        out_dir_stem = _get_dir_stem(
            parent_dirname, full_dir, include_parent=include_parent_in_stem
        )

        blob = init_blob_from_gcs_url(blob_url)
        out_dir = os.path.join(out_dir_prefix, out_dir_stem)
        download_blob_to_file(blob, out_dir, filename)


def _get_dir_stem(parent_dirname, full_dirname, include_parent=True):

    dir_components = Path(full_dirname).parts
    stem_start_idx = dir_components.index(parent_dirname)

    if not include_parent:
        stem_start_idx += 1

    stem_dir = dir_components[stem_start_idx:]

    if stem_dir:
        stem_dir = str(Path(*stem_dir))
    else:
        stem_dir = ""

    return stem_dir


def upload_dir_to_gcs(bucket_name: str, blob_prefix: str, source_dir: Path) -> None:
    """
    Uploads all files in specified directory to GCS directory
    """
    logger.info(f"Uploading timestep to gcs (blob_prefix={blob_prefix})")
    logger.debug(f"GCS bucket = {bucket_name}")
    logger.debug(f"Source local dir = {source_dir}")

    # function would not upload anything but should fail noticeably
    if not source_dir.exists():
        raise FileNotFoundError("Provided directory to upload does not exist.")

    if not source_dir.is_dir():
        raise ValueError("Provided source is not a directory.")

    upload_args = [
        (bucket_name, blob_prefix, filepath)
        for filepath in source_dir.glob("*")
        if filepath.is_file()
    ]
    upload_args_bag = db.from_sequence(upload_args)
    upload_args_bag.map(_upload_file_to_gcs).compute(scheduler="threads", num_workers=2)


def _upload_file_to_gcs(args):
    bucket_name, blob_prefix, filepath = args

    filename = filepath.name
    blob_name = blob_prefix + "/" + filename
    destination_blob = init_blob(bucket_name, blob_name)
    destination_blob.upload_from_filename(str(filepath))


def list_bucket_files(
    client: Client,
    bucket_name: str,
    prefix=None,
    file_extension=None,
    is_dir=False,
) -> Iterable[str]:

    blob_list = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blob_list:

        # filter specific extensions
        if file_extension is not None:
            blob_extension_name = blob.name.split(".")[-1]
            if file_extension.strip(".") != blob_extension_name:
                continue
        
        # Excludes top level directory that matches prefix
        if is_dir and blob.name[-1] == "/":
            continue

        yield f"gs://{blob.bucket.name}/{blob.name}"

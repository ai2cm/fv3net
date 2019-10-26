import tempfile
import logging
import subprocess
from pathlib import Path

from coarseflow.file_lister import GCSLister
from coarseflow.utils import init_blob_from_gcs_url

from google.cloud.storage import Client, Bucket, Blob


lister = GCSLister(Client(), 'vcm-ml-data')
run_dir_prefix = '2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data'
run_file_tars = lister.list(prefix=run_dir_prefix, file_extension='tar')

def _download_file(gcs_path, local_path):

    blob = init_blob_from_gcs_url(gcs_path)
    



with tempfile.TemporaryDirectory() as tmpdir:
    pass
    
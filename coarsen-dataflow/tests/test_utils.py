import pytest
import coarseflow.utils as cfutils
import hashlib
import tempfile
import os

from pathlib import Path
from google.cloud.storage import Blob

TEST_DIR = Path(os.path.abspath(__file__)).parent

def test_init_blob_is_blob():
    result = cfutils.init_blob('test_bucket', 'test_blobdir/test_blob.nc')
    assert isinstance(result, Blob)

def test_init_blob_bucket_and_blob_name():
    result = cfutils.init_blob('test_bucket', 'test_blobdir/test_blob.nc')
    assert result.bucket.name == 'test_bucket'
    assert result.name == 'test_blobdir/test_blob.nc'

def test_init_blob_from_gcs_url():
    result = cfutils.init_blob_from_gcs_url('gs://test_bucket/test_blobdir/test_blob.nc')
    assert isinstance(result, Blob)
    assert result.bucket.name == 'test_bucket'
    assert result.name == 'test_blobdir/test_blob.nc'

@pytest.mark.parametrize(
    'gcs_url', 
    ['gs://vcm-ml-data/tmp_dataflow/test_data/test_datafile.txt',
     'gs://vcm-ml-data/tmp_dataflow/test_data/test_data_array.nc']
)
def test_files_exist_on_gcs(gcs_url):
    blob = cfutils.init_blob_from_gcs_url(gcs_url)
    assert blob.exists()

def test_download_blob_to_file():
    txt_filename = 'test_datafile.txt'
    gcs_path = 'gs://vcm-ml-data/tmp_dataflow/test_data/'
    local_filepath = Path(TEST_DIR, 'test_data/test_datafile.txt')

    with tempfile.TemporaryDirectory() as tmpdir:
        blob = cfutils.init_blob_from_gcs_url(gcs_path + txt_filename)
        outfile_path = cfutils.download_blob_to_file(blob, tmpdir, txt_filename)

        assert outfile_path.exists()
        assert local_filepath.exists()

        with open(outfile_path, 'rb') as downloaded_file:
            with open(local_filepath, 'rb') as local_file:
                downloaded_file = downloaded_file.read()
                local_file = local_file.read()
                downloaded_checksum = hashlib.md5(downloaded_file).hexdigest()
                local_checksum = hashlib.md5(local_file).hexdigest()
                assert downloaded_checksum == local_checksum

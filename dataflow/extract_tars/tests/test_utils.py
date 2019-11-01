import pytest
import dataflow_utils.utils as utils
import dataflow_utils.gcs_utils as gcs_utils
import hashlib
import tempfile
import os
import shutil

from pathlib import Path
from google.cloud.storage import Blob

TEST_DIR = Path(os.path.abspath(__file__)).parent

def _compare_checksums(file_path1: Path, file_path2: Path) -> None:

    with open(file_path1, 'rb') as file1:
        with open(file_path2, 'rb') as file2:
            file1 = file1.read()
            file2 = file2.read()
            downloaded_checksum = hashlib.md5(file1).hexdigest()
            local_checksum = hashlib.md5(file2).hexdigest()
            assert downloaded_checksum == local_checksum


def test_init_blob_is_blob():
    result = gcs_utils.init_blob('test_bucket', 'test_blobdir/test_blob.nc')
    assert isinstance(result, Blob)

def test_init_blob_bucket_and_blob_name():
    result = gcs_utils.init_blob('test_bucket', 'test_blobdir/test_blob.nc')
    assert result.bucket.name == 'test_bucket'
    assert result.name == 'test_blobdir/test_blob.nc'

def test_init_blob_from_gcs_url():
    result = gcs_utils.init_blob_from_gcs_url('gs://test_bucket/test_blobdir/test_blob.nc')
    assert isinstance(result, Blob)
    assert result.bucket.name == 'test_bucket'
    assert result.name == 'test_blobdir/test_blob.nc'

@pytest.mark.parametrize(
    'gcs_url', 
    ['gs://vcm-ml-data/tmp_dataflow/test_data/test_datafile.txt',
     'gs://vcm-ml-data/tmp_dataflow/test_data/test_data_array.nc']
)
def test_files_exist_on_gcs(gcs_url):
    blob = gcs_utils.init_blob_from_gcs_url(gcs_url)
    assert blob.exists()

def test_download_blob_to_file():
    txt_filename = 'test_datafile.txt'
    gcs_path = 'gs://vcm-ml-data/tmp_dataflow/test_data/'
    local_filepath = Path(TEST_DIR, 'test_data/test_datafile.txt')

    with tempfile.TemporaryDirectory() as tmpdir:
        blob = gcs_utils.init_blob_from_gcs_url(gcs_path + txt_filename)
        outfile_path = gcs_utils.download_blob_to_file(blob, tmpdir, txt_filename)

        assert outfile_path.exists()
        assert local_filepath.exists()

        _compare_checksums(outfile_path, local_filepath)

def test_extract_tarball_default_dir():

    tar_filename = 'test_data.tar'
    test_tarball_path = Path(__file__).parent.joinpath('test_data', tar_filename)    

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
        working_path = Path(tmpdir, tar_filename)
        
        tarball_extracted_path = utils.extract_tarball_to_path(working_path)
        assert tarball_extracted_path.exists()
        assert tarball_extracted_path.name == 'test_data'

def test_extract_tarball_specified_dir():

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = 'test_data.tar'
    test_tarball_path = Path(__file__).parent.joinpath('test_data', tar_filename)    
    target_output_dirname = 'specified'

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
        target_path = Path(tmpdir, target_output_dirname)
        
        tarball_extracted_path = utils.extract_tarball_to_path(
            test_tarball_path, extract_to_dir=target_path
        )
        assert tarball_extracted_path.exists()
        assert tarball_extracted_path.name == target_output_dirname

def test_extract_tarball_check_files_exist():

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = 'test_data.tar'
    test_tarball_path = Path(__file__).parent.joinpath('test_data', tar_filename)    
    
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
        working_path = Path(tmpdir, tar_filename)
        tarball_extracted_path = utils.extract_tarball_to_path(working_path)

        test_data_files = ['test_data_array.nc', 'test_datafile.txt']
        for current_file in test_data_files:
            assert tarball_extracted_path.joinpath(current_file).exists()

def test_upload_dir_to_gcs():
    src_dir_to_upload = Path(__file__).parent.joinpath('test_data')
    gcs_utils.upload_dir_to_gcs('vcm-ml-data', 'tmp_dataflow/test_upload',
                              src_dir_to_upload)

    test_files = ['test_datafile.txt', 'test_data.tar']
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in test_files:
            gcs_url = f'gs://vcm-ml-data/tmp_dataflow/test_upload/{filename}'
            file_blob = gcs_utils.init_blob_from_gcs_url(gcs_url)
            assert file_blob.exists()

            downloaded_path = gcs_utils.download_blob_to_file(
                file_blob, 
                Path(tmpdir, 'test_uploaded'),
                filename)
            local_file = src_dir_to_upload.joinpath(filename)
            _compare_checksums(local_file, downloaded_path)
            file_blob.delete()



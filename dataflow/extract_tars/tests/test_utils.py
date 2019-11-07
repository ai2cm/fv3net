import pytest
import dataflow_utils as utils
import dataflow_utils.gcs as gcs_utils
import hashlib
import tempfile
import os
import shutil
import pickle

from pathlib import Path
from google.cloud.storage import Blob
from google.api_core.exceptions import NotFound
from subprocess import CalledProcessError

TEST_DIR = Path(os.path.abspath(__file__)).parent


@pytest.fixture(scope="function")
def tmpdir():
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield temporary_dir


def _compare_checksums(file_path1: Path, file_path2: Path) -> None:

    with open(file_path1, "rb") as file1:
        with open(file_path2, "rb") as file2:
            file1 = file1.read()
            file2 = file2.read()
            downloaded_checksum = hashlib.md5(file1).hexdigest()
            local_checksum = hashlib.md5(file2).hexdigest()
            assert downloaded_checksum == local_checksum


def test_init_blob_is_blob():
    result = gcs_utils.init_blob("test_bucket", "test_blobdir/test_blob.nc")
    assert isinstance(result, Blob)


def test_init_blob_bucket_and_blob_name():
    result = gcs_utils.init_blob("test_bucket", "test_blobdir/test_blob.nc")
    assert result.bucket.name == "test_bucket"
    assert result.name == "test_blobdir/test_blob.nc"


def test_init_blob_from_gcs_url():
    result = gcs_utils.init_blob_from_gcs_url(
        "gs://test_bucket/test_blobdir/test_blob.nc"
    )
    assert isinstance(result, Blob)
    assert result.bucket.name == "test_bucket"
    assert result.name == "test_blobdir/test_blob.nc"


@pytest.mark.parametrize(
    "gcs_url",
    [
        "gs://vcm-ml-data/tmp_dataflow/test_data/test_datafile.txt",
        "gs://vcm-ml-data/tmp_dataflow/test_data/test_data_array.nc",
    ],
)
def test_files_exist_on_gcs(gcs_url):
    blob = gcs_utils.init_blob_from_gcs_url(gcs_url)
    assert blob.exists()


def test_download_blob_to_file(tmpdir):
    txt_filename = "test_datafile.txt"
    gcs_path = "gs://vcm-ml-data/tmp_dataflow/test_data/"
    local_filepath = Path(TEST_DIR, "test_data/test_datafile.txt")

    blob = gcs_utils.init_blob_from_gcs_url(gcs_path + txt_filename)
    outfile_path = gcs_utils.download_blob_to_file(blob, tmpdir, txt_filename)

    assert outfile_path.exists()
    assert local_filepath.exists()

    _compare_checksums(outfile_path, local_filepath)


def test_download_blob_to_file_makes_destination_directories(tmpdir):
    txt_filename = "test_datafile.txt"
    gcs_path = "gs://vcm-ml-data/tmp_dataflow/test_data/"
    nonexistent_path = Path("does/not/exist")

    blob = gcs_utils.init_blob_from_gcs_url(gcs_path + txt_filename)

    non_existent_dir = Path(tmpdir, nonexistent_path)
    assert not non_existent_dir.exists()

    gcs_utils.download_blob_to_file(blob, non_existent_dir, txt_filename)
    assert non_existent_dir.exists()


def test_download_glob_to_file_nonexistent_blob(tmpdir):
    nonexistent_gcs_path = "gs://vcm-ml-data/non_existent_dir/non_existent_file.lol"
    blob = gcs_utils.init_blob_from_gcs_url(nonexistent_gcs_path)

    with pytest.raises(NotFound):
        gcs_utils.download_blob_to_file(blob, tmpdir, "non_existsent.file")


def test_extract_tarball_default_dir(tmpdir):

    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    working_path = Path(tmpdir, tar_filename)

    tarball_extracted_path = utils.extract_tarball_to_path(working_path)
    assert tarball_extracted_path.exists()
    assert tarball_extracted_path.name == "test_data"


def test_extract_tarball_specified_dir(tmpdir):

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)
    target_output_dirname = "specified"

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    target_path = Path(tmpdir, target_output_dirname)

    tarball_extracted_path = utils.extract_tarball_to_path(
        test_tarball_path, extract_to_dir=target_path
    )
    assert tarball_extracted_path.exists()
    assert tarball_extracted_path.name == target_output_dirname


def test_extract_tarball_check_files_exist(tmpdir):

    # TODO: could probably create fixture for tar file setup/cleanup
    tar_filename = "test_data.tar"
    test_tarball_path = Path(__file__).parent.joinpath("test_data", tar_filename)

    shutil.copyfile(test_tarball_path, Path(tmpdir, tar_filename))
    working_path = Path(tmpdir, tar_filename)
    tarball_extracted_path = utils.extract_tarball_to_path(working_path)

    test_data_files = ["test_data_array.nc", "test_datafile.txt"]
    for current_file in test_data_files:
        assert tarball_extracted_path.joinpath(current_file).exists()


def test_extract_tarball_non_existent_tar(tmpdir):
    non_existent_tar = Path(tmpdir, "nonexistent/tarfile.tar")
    with pytest.raises(CalledProcessError):
        utils.extract_tarball_to_path(non_existent_tar)


def test_upload_dir_to_gcs(tmpdir):
    src_dir_to_upload = Path(__file__).parent.joinpath("test_data")
    gcs_utils.upload_dir_to_gcs(
        "vcm-ml-data", "tmp_dataflow/test_upload", src_dir_to_upload
    )

    test_files = ["test_datafile.txt", "test_data.tar"]

    for filename in test_files:
        gcs_url = f"gs://vcm-ml-data/tmp_dataflow/test_upload/{filename}"
        file_blob = gcs_utils.init_blob_from_gcs_url(gcs_url)
        assert file_blob.exists()

        downloaded_path = gcs_utils.download_blob_to_file(
            file_blob, Path(tmpdir, "test_uploaded"), filename
        )
        local_file = src_dir_to_upload.joinpath(filename)
        _compare_checksums(local_file, downloaded_path)
        file_blob.delete()


def test_upload_dir_to_gcs_from_nonexistent_dir(tmpdir):

    nonexistent_dir = Path(tmpdir, "non/existent/dir/")
    with pytest.raises(FileNotFoundError):
        gcs_utils.upload_dir_to_gcs(
            "vcm-ml-data", "tmp_dataflow/test_upload", nonexistent_dir
        )


def test_upload_dir_to_gcs_dir_is_file():

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            gcs_utils.upload_dir_to_gcs(
                "vcm-ml-data", "tmp_dataflow/test_upload", Path(f.name)
            )


def test_upload_dir_to_gcs_does_not_upload_subdir(tmpdir):

    x = (1, 2, 3, 4)
    with open(Path(tmpdir, "what_a_pickle.pkl"), "wb") as f:
        pickle.dump(x, f)

    extra_subdir = Path(tmpdir, "extra_dir")
    extra_subdir.mkdir()

    with open(Path(extra_subdir, "extra_pickle.pkl"), "wb") as f:
        pickle.dump(x, f)

    # TODO: use pytest fixture to do setup/teardown of temporary gcs dir

    upload_dir = "transient"
    bucket_name = "vcm-ml-data"
    gcs_url_prefix = f"gs://{bucket_name}"
    tmp_gcs_dir = f"tmp_dataflow/test_upload/{upload_dir}"
    tmp_gcs_url = f"{gcs_url_prefix}/{tmp_gcs_dir}"

    gcs_utils.upload_dir_to_gcs(bucket_name, tmp_gcs_dir, Path(tmpdir))

    uploaded_pickle_url = f"{tmp_gcs_url}/what_a_pickle.pkl"
    not_uploaded_pickle_url = f"{tmp_gcs_url}/extra_dir/extra_pickle.pkl"

    pkl_blob = gcs_utils.init_blob_from_gcs_url(uploaded_pickle_url)
    nonexistent_pkl_blob = gcs_utils.init_blob_from_gcs_url(not_uploaded_pickle_url)

    assert pkl_blob.exists()
    pkl_blob.delete()

    assert not nonexistent_pkl_blob.exists()

from typing import Iterator, Iterable

from google.cloud.storage import Client, Blob, Bucket  # type: ignore
import pytest

from dataflow_utils.gcs import list_gcs_bucket_files

class FakeFileListerClient(Client):
    def __init__(self, object_keys: Iterable[str]):
        self.object_keys = object_keys

    def get_bucket(self, name):
        return self.bucket

    def list_blobs(
            self,
            bucket_or_name,
            max_results=None,
            page_token=None,
            prefix=None,
            delimiter=None,
            versions=None,
            projection="noAcl",
            fields=None,
    ) -> Iterator[Blob]:
        bucket = Bucket(self, name=bucket_or_name)
        for key in self.object_keys:

            # check that blob starts with correct prefix
            if prefix is not None:
                prefix_idx = key.find(prefix)
                if prefix_idx != 0:
                    continue
            
            yield Blob(key, bucket)


def test_gcs_lister_all_items():
    bucket_name = 'test_bucket'
    blob_names = ['subdir/blob1.nc',
                  'diff_subdir/new_blob.xyz',
                  'middle_blob.tar']

    fake_client = FakeFileListerClient(blob_names)
    expected = [f'gs://{bucket_name}/{bname}' for bname in blob_names]

    result = list(list_gcs_bucket_files(fake_client, bucket_name))
    assert result == expected


def test_gcs_lister_use_prefix():
    bucket_name = 'test_bucket'
    blob_names = ['subdir/blob1.nc',
                  'subdir/blob2.nc',
                  'subdir2/diff_blob.tar',  
                  'diff_subdir/new_blob.xyz',
                  'middle_blob.tar']
    
    subdir_blob_names = ['subdir/blob1.nc',
                         'subdir/blob2.nc']
    prefix = 'subdir/'

    fake_client = FakeFileListerClient(blob_names)
    expected = [f'gs://{bucket_name}/{bname}' for bname in subdir_blob_names]

    result = list_gcs_bucket_files(fake_client, bucket_name, prefix=prefix)
    assert list(result) == expected


def test_gcs_lister_file_ext():
    bucket_name = 'test_bucket'
    blob_names = ['subdir/blob1.nc',
                  'subdir/blob2.nc',
                  'subdir2/diff_blob.tar',  
                  'diff_subdir/new_blob.xyz',
                  'middle_blob.tar']
    
    file_ext_blob_names = ['subdir2/diff_blob.tar',
                           'middle_blob.tar']
    file_ext = '.tar'

    fake_client = FakeFileListerClient(blob_names)
    expected = [f'gs://{bucket_name}/{bname}' for bname in file_ext_blob_names]

    result = list_gcs_bucket_files(fake_client, bucket_name, file_extension=file_ext)
    assert list(result) == expected


def test_gcs_lister_file_ext_and_prefix():
    bucket_name = 'test_bucket'
    blob_names = ['subdir/blob1.nc',
                  'subdir/blob2.nc',
                  'subdir2/diff_blob.tar'  
                  'diff_subdir/new_blob.xyz',
                  'middle_blob.tar',
                  'greater_blob.nc']
    
    file_ext_blob_names = ['subdir/blob1.nc',
                           'subdir/blob2.nc']
    file_ext = '.nc'
    prefix = 'subdir/'

    fake_client = FakeFileListerClient(blob_names)
    expected = [f'gs://{bucket_name}/{bname}' for bname in file_ext_blob_names]

    result = list_gcs_bucket_files(fake_client, bucket_name, 
                                   file_extension=file_ext,
                                   prefix=prefix)
    assert list(result) == expected


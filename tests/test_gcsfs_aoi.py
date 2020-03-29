from gcsfs_aoi import __version__, GCSFSMapperAoi
import pytest


def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize('url, bucket', [
    ("gs://vcm-ml-data//hello/a.zarr", "vcm-ml-data"),
    ("gs://fun-stuff/hello//a.zarr/b", "fun-stuff"),
])
def test_mapper_bucket(url, bucket):
    mapper = GCSFSMapperAoi(url)
    assert mapper.bucket == bucket


@pytest.mark.parametrize('url, prefix', [
    ("gs://vcm-ml-data/hello/a.zarr", "hello/a.zarr"),
    ("gs://vcm-ml-data/hello//a.zarr/b", "hello//a.zarr/b"),
])
def test_mapper_prefix(url, prefix):
    mapper = GCSFSMapperAoi(url)
    assert mapper.prefix == prefix
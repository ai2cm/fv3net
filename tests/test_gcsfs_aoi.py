from gcs_aio_mapper import __version__, GCSMapperAio
import pytest


def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize('url, bucket', [
    ("gs://vcm-ml-data//hello/a.zarr", "vcm-ml-data"),
    ("gs://fun-stuff/hello//a.zarr/b", "fun-stuff"),
])
def test_mapper_bucket(url, bucket):
    mapper = GCSMapperAio(url)
    assert mapper.bucket == bucket


@pytest.mark.parametrize('url, prefix', [
    ("gs://vcm-ml-data/hello/a.zarr", "hello/a.zarr"),
    ("gs://vcm-ml-data/hello//a.zarr/b", "hello//a.zarr/b"),
])
def test_mapper_prefix(url, prefix):
    mapper = GCSMapperAio(url)
    assert mapper.prefix == prefix


def test_set():
    map = GCSMapperAio('gs://vcm-ml-data/tmp/test1.zarr')
    map['0'] = b"123"
    map['1'] = b"234"
    map.flush()


def test_get():
    map = GCSMapperAio('gs://vcm-ml-data/tmp/test2.zarr')
    map['0'] = b"123"
    map['1'] = b"234"
    del map

    map = GCSMapperAio('gs://vcm-ml-data/tmp/test2.zarr')
    assert map['1'] == b"234"

def test_set_error():
    map = GCSMapperAio('gs://a-non-existant-bucket/test.zarr')
    map['0'] = b"123"
    map['1'] = b"234"
    with pytest.raises(Exception):
        map.flush()
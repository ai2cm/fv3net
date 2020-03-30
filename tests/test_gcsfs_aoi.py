from gcs_aio_mapper import __version__, GCSMapperAio
import pytest


TEST_BUCKET = "vcm-ml-data"


def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize('url, bucket', [
    (f"gs://{TEST_BUCKET}//hello/a.zarr", f"{TEST_BUCKET}"),
    ("gs://fun-stuff/hello//a.zarr/b", "fun-stuff"),
])
def test_mapper_bucket(url, bucket):
    mapper = GCSMapperAio(url)
    assert mapper.bucket == bucket


@pytest.mark.parametrize('url, prefix', [
    ("gs://{TEST_BUCKET}/hello/a.zarr", "hello/a.zarr"),
    ("gs://{TEST_BUCKET}/hello//a.zarr/b", "hello//a.zarr/b"),
])
def test_mapper_prefix(url, prefix):
    mapper = GCSMapperAio(url)
    assert mapper.prefix == prefix


def test_set():
    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test1.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    mapper.flush()


def test_get():
    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test2.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    del mapper

    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test2.zarr')
    assert mapper['1'] == b"234"

def test_set_error():
    mapper = GCSMapperAio('gs://a-non-existant-bucket/test.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    with pytest.raises(Exception):
        mapper.flush()


def test_rmdir_root():
    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test2.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    mapper.flush()
    mapper['2'] = b"234"
    mapper.rmdir()
    assert set(mapper.keys()) == set()

def test_rmdir_path():
    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test2.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    mapper['base/2'] = b"234"
    mapper['base/3'] = b"234"
    mapper.flush()
    mapper.rmdir(path='base')
    assert set(mapper.keys()) == {'0', '1'}

def test_keys():
    mapper = GCSMapperAio(f'gs://{TEST_BUCKET}/tmp/test2.zarr')
    mapper['0'] = b"123"
    mapper['1'] = b"234"
    mapper['.zarray'] = b"234"
    mapper.flush()
    mapper['3'] = b"234"
    assert set(mapper.keys()) == {'0', '1', '3', '.zarray'}
import fsspec
from vcm.cloud.fsspec import get_protocol, copy, to_url

import pytest


@pytest.mark.parametrize(
    ("path", "expected_protocol"),
    [
        ("gs://bucket/path/to/file", "gs"),
        ("/absolute/file/path", "file"),
        ("relative/file/path", "file"),
    ],
)
def test__get_protocol(path, expected_protocol):
    assert get_protocol(path) == expected_protocol


@pytest.mark.parametrize(
    ("content_type"), [None, "text/html"],
)
def test_copy(tmpdir, content_type):
    expected_content = "foobar"
    source = tmpdir.join("file1.txt")
    destination = tmpdir.join("file2.txt")
    source.write(expected_content)
    copy(str(source), str(destination), content_type=content_type)
    with open(destination) as f:
        content = f.read()
    assert content == expected_content


@pytest.mark.parametrize(
    "protocol,path,expected",
    [
        ("gs", "some-path", "gs://some-path"),
        ("file", "some-path", "file://some-path"),
        ("http", "some-path", "http://some-path"),
    ],
)
def test_to_url(protocol, path, expected):
    assert expected == to_url(fsspec.filesystem(protocol), path)

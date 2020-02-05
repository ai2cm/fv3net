from vcm.cloud.fsspec import get_protocol

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

import subprocess

import pytest

from subprocess import CalledProcessError
from unittest.mock import MagicMock
from fv3post.gsutil import upload_dir, GSUtilResumableUploadException


UPLOAD_EXCEPTION_STR = CalledProcessError(-1, None, "foo\nResumableUploadException:")
UPLOAD_EXCEPTION_BYTES = CalledProcessError(-1, None, b"foo\nResumableUploadException:")
OTHER_EXCEPTION = CalledProcessError(-1, None, output=b"Other")


@pytest.mark.parametrize(
    ("side_effect", "exception", "call_count"),
    [
        (UPLOAD_EXCEPTION_STR, GSUtilResumableUploadException, 3),
        (UPLOAD_EXCEPTION_BYTES, GSUtilResumableUploadException, 3),
        (OTHER_EXCEPTION, CalledProcessError, 1),
        (None, None, 1),
        ([UPLOAD_EXCEPTION_STR, None], None, 2),
        ([UPLOAD_EXCEPTION_BYTES, None], None, 2),
    ],
)
def test_upload_dir(monkeypatch, side_effect, exception, call_count):
    mock = MagicMock(side_effect=side_effect)
    monkeypatch.setattr(subprocess, "check_output", mock)
    if exception is not None:
        with pytest.raises(exception):
            upload_dir("foo", "bar")
    else:
        upload_dir("foo", "bar")
    assert mock.call_count == call_count

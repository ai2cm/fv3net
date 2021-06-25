import datetime
from fv3net.artifacts.resolve_url import resolve_url
import pytest


def test_resolve_url_raises_value_error():
    with pytest.raises(ValueError):
        resolve_url("my/bucket", "project", "tag")
    with pytest.raises(ValueError):
        resolve_url("bucket", "my/project", "tag")
    with pytest.raises(ValueError):
        resolve_url("bucket", "project", "my/tag")


def test_resolve_url(monkeypatch):
    class mydatetime:
        FAKE_TIME = datetime.datetime(2021, 6, 3, 3, 2, 4)

        @classmethod
        def now(cls):
            return cls.FAKE_TIME

    monkeypatch.setattr(datetime, "datetime", mydatetime)

    result = resolve_url("vcm-ml", "default", "clever-experiment")
    expected_result = "gs://vcm-ml/default/2021-06-03/clever-experiment"
    assert result == expected_result

import datetime
from fv3net.artifacts.generate import generate
import pytest

FAKE_TIME = datetime.datetime(2021, 6, 3, 3, 2, 4)


def test_generate_raises_value_error():
    with pytest.raises(ValueError):
        generate("my/bucket", "project", "tag")
    with pytest.raises(ValueError):
        generate("bucket", "my/project", "tag")
    with pytest.raises(ValueError):
        generate("bucket", "project", "my/tag")


def test_generate(monkeypatch):
    class mydatetime:
        @classmethod
        def now(cls):
            return FAKE_TIME

    monkeypatch.setattr(datetime, "datetime", mydatetime)

    result = generate("vcm-ml", "default", "clever-experiment")
    expected_result = "gs://vcm-ml/default/2021-06-03/clever-experiment"
    assert result == expected_result

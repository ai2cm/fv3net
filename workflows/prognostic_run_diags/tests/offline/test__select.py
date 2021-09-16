import pytest
from offline_ml_diags._select import nearest_time


@pytest.mark.parametrize(
    "select_time, closest_key",
    [
        ("20160801.010000", "20160801.000000"),
        ("20160801.200000", "20160801.180000"),
        ("20190801.000000", "20170801.180000"),
    ],
)
def test_nearest_time(select_time, closest_key):
    times = ["20160801.000000", "20160801.180000", "20170801.180000"]
    assert nearest_time(select_time, times) == closest_key

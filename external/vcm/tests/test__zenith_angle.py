from datetime import datetime
import pytest

from vcm._zenith_angle import cos_zenith_angle


@pytest.mark.parametrize(
    "time, lon, lat, expected",
    (
        [datetime(2020, 7, 6, 0, 0, 0), 90., 0., -0.019297],
        [datetime(2020, 7, 6, 12, 0, 0), -90., 0., -0.0196310],
        [datetime(2020, 7, 6, 9, 0, 0), 40., 40., 0.9501915],
        [datetime(2020, 7, 6, 12, 0, 0), 0., 90., 0.3843733],

    )
)
def test__sun_zenith_angle(time, lon, lat, expected):
    assert cos_zenith_angle(time, lon, lat) == pytest.approx(expected, abs=1e-3)
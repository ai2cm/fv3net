import pytest
import cftime
from fv3net.diagnostics.offline._select import nearest_time_batch_index


BATCH_ONE = [
    cftime.DatetimeJulian(2016, 8, 1, 11),
    cftime.DatetimeJulian(2016, 8, 1, 14),
]
BATCH_TWO = [
    cftime.DatetimeJulian(2016, 8, 1, 17),
    cftime.DatetimeJulian(2016, 8, 1, 20),
]
BATCH_THREE = [
    cftime.DatetimeJulian(2016, 8, 1, 11, 30),
    cftime.DatetimeJulian(2016, 8, 1, 20),
]


@pytest.mark.parametrize(
    "time, time_batches, expected_index",
    [
        (cftime.DatetimeJulian(2016, 8, 1, 12), [BATCH_ONE], 0,),
        (cftime.DatetimeJulian(2016, 8, 1, 12), [BATCH_ONE, BATCH_TWO], 0,),
        (cftime.DatetimeJulian(2016, 8, 1, 12), [BATCH_ONE, BATCH_THREE], 1,),
    ],
)
def test_nearest_time(time, time_batches, expected_index):
    index = nearest_time_batch_index(time, time_batches)
    assert index == expected_index

import xarray
import cftime
import numpy
import json
import vcm


def dump_times(times, f):
    json.dump([vcm.convenience.encode_time(time) for time in times], f)


numpy.random.seed(0)
ds = xarray.open_zarr(
    "/Users/noah/data/gs/vcm-ml-archive/noahb/hybrid-fine-res/2021-05-05-hybrid-training.zarr"  # noqa: E501
)


time = ds.time.values
train_period = time[
    (time > cftime.DatetimeJulian(2016, 8, 5))
    & (time <= cftime.DatetimeJulian(2016, 9, 1))
]
test_period = time[(time > cftime.DatetimeJulian(2016, 9, 1))]

test_samples = numpy.random.choice(test_period, 60)
train_samples = numpy.random.choice(train_period, 130)

with open("train.json", "w") as f:
    dump_times(train_samples, f)

with open("test.json", "w") as f:
    dump_times(test_samples, f)

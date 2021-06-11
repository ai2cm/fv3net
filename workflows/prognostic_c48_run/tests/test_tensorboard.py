from unittest.mock import Mock

import cftime
import numpy as np
import tensorflow as tf
import xarray
from runtime.diagnostics.tensorboard import TensorBoardSink


def test_tensorboardsink(monkeypatch):
    state = xarray.Dataset({"a": (["y", "x"], np.ones((10, 10)))})
    time = cftime.DatetimeJulian(2000, 1, 1)
    sink = TensorBoardSink()
    image_summary = Mock()
    monkeypatch.setattr(tf.summary, "image", image_summary)
    sink.sink(time, state)

    assert image_summary.called

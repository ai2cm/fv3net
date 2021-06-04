import io
import logging
from typing import Mapping

import cftime
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr

logger = logging.getLogger(__name__)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class TensorBoardSink:
    def __init__(self):
        self.step = 0

    def sink(self, time: cftime.DatetimeJulian, data: Mapping[str, xr.DataArray]):
        for variable in data:
            fig = plt.figure()
            data[variable].plot()
            tf.summary.image(f"{variable}", plot_to_image(fig), step=self.step)
            plt.close(fig)

        self.step += 1

import logging
from typing import Mapping

import cftime
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr

from fv3fit.tensorboard import plot_to_image

logger = logging.getLogger(__name__)


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

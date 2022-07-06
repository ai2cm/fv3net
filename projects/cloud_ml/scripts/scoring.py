import os
import fv3viz
import xarray as xr
import numpy as np
from vcm.calc.calc import weighted_average
from vcm.calc import metrics
from vcm import mass_integrate


def find_common_times(ds1, ds2):
    return list(set(ds1.time.values).intersection(set(ds2.time.values)))


def global_average(x):
    return x.mean(["x", "y", "tile"])


def r2_score(x, y, mean=global_average):
    """By default leaves time and z

    This is good for profiles

    """
    var = mean((x - mean(x)) ** 2)
    mse = metrics.mean_squared_error(x, y, mean=mean)
    return 1 - mse / var


def r2_score_grid_point(x, y, mean=lambda x: x.mean()):
    """This should be equivalent to the r2_score function in sklearn

    This is good for the grid-point problem

    """
    var = mean((x - mean(x)) ** 2)
    mse = metrics.mean_squared_error(x, y, mean=mean)
    return 1 - mse / var

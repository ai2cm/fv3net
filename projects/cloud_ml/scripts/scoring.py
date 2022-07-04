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
    return x.mean(['x', 'y','tile'])

def global_sum(x):
    return x.sum(['x', 'y','tile'])

def global_variance(x):
    ## returns the errors relative its global mean
    return x.var(['x', 'y','tile'])

def r2_score(x, y, mean = global_average):
    var  = mean((x - mean(x))**2)
    mse  = metrics.mean_squared_error(x, y, mean = mean)
    return 1 - mse/var 

def r2_integrated_skill(x, y, delp, z_dim = 'pressure',mean = global_average):
    mean_integrated_variance  = mean(mass_integrate((x - mean(x))**2 , delp, z_dim))
    mean_integrated_errors  = mean(mass_integrate((x -y)**2 , delp, z_dim))
    return 1 - mean_integrated_errors / mean_integrated_variance

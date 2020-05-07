import numpy as np
import xarray as xr
from vcm.convenience import round_time

time = xr.open_dataset('time.nc').time

def round_time_fails(x):
    try:
        round_time(x)
        return False
    except:
        return True
    

broken_times =  np.vectorize(round_time_fails)(time)
import numpy as np
from config import *
import os 
import xarray as xr 

def read_lookupdata(LOOKUP_DIR, me):
    
    data_dict = {}
    # File names for serialized random numbers in mcica_subcol
    sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_sw.nc")
    lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_lw.nc")
    print('sw random file: ' + sw_rand_file)
    print('lw random file: ' + lw_rand_file)
    data_dict['sw_rand'] = xr.open_dataset(sw_rand_file)['rand2d'].values
    data_dict['lw_rand'] = xr.open_dataset(lw_rand_file)['rand2d'].values

    return data_dict 


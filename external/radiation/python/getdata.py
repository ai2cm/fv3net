import numpy as np
from config import *
import os 
import xarray as xr 


def random_numbers(LOOKUP_DIR, me):
    data_dict = {}
    # File names for serialized random numbers in mcica_subcol
    sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_sw.nc")
    lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_lw.nc")
    data_dict['sw_rand'] = xr.open_dataset(sw_rand_file)['rand2d'].values
    data_dict['lw_rand'] = xr.open_dataset(lw_rand_file)['rand2d'].values
    del(sw_rand_file, lw_rand_file)

    return data_dict 



def lw(LOOKUP_DIR):
        ## file names needed in lwrad()
    
    longwave_dict = {}
    dfile = os.path.join(LOOKUP_DIR, "totplnk.nc")
    pfile = os.path.join(LOOKUP_DIR, "radlw_ref_data.nc")
    longwave_dict['totplnk'] = xr.open_dataset(dfile)["totplnk"].values
    longwave_dict['preflog'] = xr.open_dataset(pfile)["preflog"].values
    longwave_dict['tref'] = xr.open_dataset(pfile)["tref"].values
    longwave_dict['chi_mls'] = xr.open_dataset(pfile)["chi_mls"].values
    del(dfile, pfile)


    ## loading data for cldprop in lwrad()
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_cldprlw_data.nc"))
    longwave_dict['absliq1'] = ds["absliq1"].values
    longwave_dict['absice0'] = ds["absice0"].values
    longwave_dict['absice1'] = ds["absice1"].values
    longwave_dict['absice2'] = ds["absice2"].values
    longwave_dict['absice3'] = ds["absice3"].values
    del(ds)

    return longwave_dict


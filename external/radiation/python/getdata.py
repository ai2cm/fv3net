import numpy as np
from config import *
import os 
import xarray as xr 

def read_lookupdata(LOOKUP_DIR, me):
    
    data_dict = {}
    # File names for serialized random numbers in mcica_subcol
    sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_sw.nc")
    lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_lw.nc")

    data_dict['sw_rand'] = xr.open_dataset(sw_rand_file)['rand2d'].values
    data_dict['lw_rand'] = xr.open_dataset(lw_rand_file)['rand2d'].values
    del(sw_rand_file, lw_rand_file)

    ## file names needed in lwrad()
    dfile = os.path.join(LOOKUP_DIR, "totplnk.nc")
    pfile = os.path.join(LOOKUP_DIR, "radlw_ref_data.nc")
    data_dict['totplnk'] = xr.open_dataset(dfile)["totplnk"].values
    data_dict['preflog'] = xr.open_dataset(pfile)["preflog"].values
    data_dict['tref'] = xr.open_dataset(pfile)["tref"].values
    data_dict['chi_mls'] = xr.open_dataset(pfile)["chi_mls"].values
    del(dfile, pfile)
    ## loading data for cldprop in lwrad()
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_cldprlw_data.nc"))
    data_dict['absliq1-lw'] = ds["absliq1"].values
    data_dict['absice0-lw'] = ds["absice0"].values
    data_dict['absice1-lw'] = ds["absice1"].values
    data_dict['absice2-lw'] = ds["absice2"].values
    data_dict['absice3-lw'] = ds["absice3"].values
    del(ds)

    return data_dict 


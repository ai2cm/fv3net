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

    ## loading data for taumol
    ds_bands = {}
    for nband in range(1,17):
        if nband < 10:
            ds_bands['radlw_kgb0' + str(nband)] = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb0" + str(nband) + "_data.nc"))
        else:
            ds_bands['radlw_kgb' + str(nband)] = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb" + str(nband) + "_data.nc"))
    bands = {}
    varnames_bands = {1:['selfref','forref','ka_mn2','absa','absb','fracrefa','fracrefb'],
                      2:['selfref','forref','absa','absb','fracrefa','fracrefb'],
                      3:['selfref','forref','ka_mn2o','kb_mn2o','absa','absb','fracrefa','fracrefb'],
                      4:['selfref','forref','absa','absb','fracrefa','fracrefb'],
                      5:['selfref','forref','absa','absb','fracrefa','fracrefb','ka_mo3','ccl4'],
                      6:['selfref','forref','absa','fracrefa','ka_mco2','cfc11adj','cfc12'],
                      7:['selfref','forref','absa','absb','fracrefa','fracrefb','ka_mco2','kb_mco2'],
                      8:['selfref','forref','absa','absb','fracrefa','fracrefb','ka_mo3','ka_mco2','kb_mco2','cfc12','ka_mn2o','kb_mn2o','cfc22adj'],
                      9:['selfref','forref','absa','absb','fracrefa','fracrefb','ka_mn2o','kb_mn2o'],
                      10:['selfref','forref','absa','absb','fracrefa','fracrefb'],
                      11:['selfref','forref','absa','absb','fracrefa','fracrefb','ka_mo2','kb_mo2'],
                      12:['selfref','forref','absa','fracrefa'],
                      13:['selfref','forref','absa','fracrefa','fracrefb','ka_mco2','ka_mco','kb_mo3'],
                      14:['selfref','forref','absa','absb','fracrefa','fracrefb'],
                      15:['selfref','forref','absa','fracrefa','ka_mn2'],
                      16:['selfref','forref','absa','absb','fracrefa','fracrefb'],
                    }

    for nband in range(1, 17): 
        print(nband)
        if nband < 10:
            data =  ds_bands['radlw_kgb0' + str(nband)]
        else:
            data =  ds_bands['radlw_kgb' + str(nband)]
        tmp = {}
        for var in varnames_bands[nband]:
            tmp[var] =data[var].values
        longwave_dict['band' + str(nband)] = tmp

    return longwave_dict


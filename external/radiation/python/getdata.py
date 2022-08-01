import numpy as np
from config import *
import os 
import xarray as xr 
import warnings


def random_numbers(LOOKUP_DIR, me):
    data_dict = {}
    # File names for serialized random numbers in mcica_subcol
    if me == 0:
        sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_sw.nc")
    else:
        sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_sw.nc")
    lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile" + str(me) + "_lw.nc")
    data_dict['sw_rand'] = xr.open_dataset(sw_rand_file)['rand2d'].values
    data_dict['lw_rand'] = xr.open_dataset(lw_rand_file)['rand2d'].values
    del(sw_rand_file, lw_rand_file)

    return data_dict 

def lw(LOOKUP_DIR):
        ## file names needed in lwrad()
    
    lw_dict = {}
    dfile = os.path.join(LOOKUP_DIR, "totplnk.nc")
    pfile = os.path.join(LOOKUP_DIR, "radlw_ref_data.nc")
    lw_dict['totplnk'] = xr.open_dataset(dfile)["totplnk"].values
    lw_dict['preflog'] = xr.open_dataset(pfile)["preflog"].values
    lw_dict['tref'] = xr.open_dataset(pfile)["tref"].values
    lw_dict['chi_mls'] = xr.open_dataset(pfile)["chi_mls"].values
    del(dfile, pfile)


    ## loading data for cldprop in lwrad()
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_cldprlw_data.nc"))
    lw_dict['absliq1'] = ds["absliq1"].values
    lw_dict['absice0'] = ds["absice0"].values
    lw_dict['absice1'] = ds["absice1"].values
    lw_dict['absice2'] = ds["absice2"].values
    lw_dict['absice3'] = ds["absice3"].values
    del(ds)

    ## loading data for taumol
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
        if nband < 10:
            data =  xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb0" + str(nband) + "_data.nc"))
        else:
            data =  xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb" + str(nband) + "_data.nc"))
        tmp = {}
        for var in varnames_bands[nband]:
            tmp[var] =data[var].values
        lw_dict['band' + str(nband)] = tmp

    return lw_dict

def sw(LOOKUP_DIR):
    sw_dict = {}
    
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_sflux_data.nc"))
    sw_dict['strrat'] = ds["strrat"].values
    sw_dict['specwt'] = ds["specwt"].values
    sw_dict['layreffr'] = ds["layreffr"].values
    sw_dict['ix1'] = ds["ix1"].values
    sw_dict['ix2'] = ds["ix2"].values
    sw_dict['ibx'] = ds["ibx"].values
    sw_dict['sfluxref01']  = ds["sfluxref01"].values
    sw_dict['sfluxref02']  = ds["sfluxref02"].values
    sw_dict['sfluxref03']  = ds["sfluxref03"].values
    sw_dict['scalekur']  = ds["scalekur"].values
    del(ds)
    ## data loading for setcoef
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_ref_data.nc"))
    sw_dict['preflog'] = ds["preflog"].values
    sw_dict['tref'] = ds["tref"].values
    del(ds)
    ## load data for cldprop
    ds_cldprtb = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_cldprtb_data.nc"))
    var_names = ['extliq1','extliq2','ssaliq1','ssaliq2',
    'asyliq1','asyliq2','extice2','ssaice2','asyice2',
    'extice3','ssaice3','asyice3','abari','bbari',
    'cbari','dbari','ebari','fbari','b0s','b1s','b0r',
    'b0r','c0s','c0r','a0r','a1r','a0s','a1s']

    for var in var_names:
        sw_dict[var] =  ds_cldprtb[var].values
    del(ds_cldprtb)
    
    ## loading data for taumol
    varnames_bands = {16:['selfref','forref','absa','absb','rayl'],
                    17:['selfref','forref','absa','absb','rayl'],
                    18:['selfref','forref','absa','absb','rayl'],
                    19:['selfref','forref','absa','absb','rayl'],
                    20:['selfref','forref','absa','absb','absch4','rayl'],
                    21:['selfref','forref','absa','absb','rayl'],
                    22:['selfref','forref','absa','absb','rayl'],
                    23:['selfref','forref','absa','rayl','givfac'],
                    24:['selfref','forref','absa','absb','abso3a','abso3b','rayla','raylb'],
                    25:['absa','abso3a','abso3b','rayl'],
                    26:['rayl'],
                    27:['absa','absb','rayl'],
                    28:['absa','absb','rayl'],
                    29:['forref','absa','absb','selfref','absh2o','absco2','rayl']
                    }
                  
    for nband in range(16, 30): 
        data =  xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb" + str(nband) + "_data.nc"))
        tmp = {}
        for var in varnames_bands[nband]:
            tmp[var] =data[var].values
        sw_dict['band' + str(nband)] = tmp

    return sw_dict

def aerosol(FORCING_DIR):
    aeros_file = os.path.join(FORCING_DIR, 'aerosol.nc')
    if os.path.isfile(aeros_file):
        print(f"Using file {aeros_file}")
    else:
        raise FileNotFoundError(
            f'Requested aerosol data file "{aeros_file}" not found!',
            "*** Stopped in subroutine aero_init !!",
            )
    var_names= ['kprfg','kprfg','idxcg','cmixg','denng','cline',
                'iendwv','haer','prsref','rhidext0','rhidsca0',
                'rhidssa0','rhidasy0','rhdpext0','rhdpsca0',
                'rhdpssa0','rhdpasy0','straext0']
    data_dict = {}
    ds = xr.open_dataset(aeros_file)
    for var in var_names:
        data_dict[var] = ds[var].values

    return data_dict

def astronomy(FORCING_DIR, isolar, me):
    # external solar constant data table,solarconstant_noaa_a0.txt

    if me == 0:
        if isolar == 1: # noaa ann-mean tsi in absolute scale
            solar_file = "solarconstant_noaa_a0.nc"

            if os.path.isfile(os.path.join(FORCING_DIR, solar_file)):
                data = xr.open_dataset(os.path.join(FORCING_DIR, solar_file))
            else:
                warnings.warn(
                            f'Requested solar data file "{solar_file}" not found!',
                        )
                raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!")

        elif isolar == 2:# noaa ann-mean tsi in tim scale
            solar_file = "solarconstant_noaa_an.nc"
            if os.path.isfile(os.path.join(FORCING_DIR, solar_file)):
                data = xr.open_dataset(os.path.join(FORCING_DIR, solar_file))
            else:
                warnings.warn(
                            f'Requested solar data file "{solar_file}" not found!',
                        )
                raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!")

        elif isolar == 3:# cmip5 ann-mean tsi in tim scale
            solar_file ='solarconstant_cmip_an.nc'
            if os.path.isfile(os.path.join(FORCING_DIR, solar_file)):
                data = xr.open_dataset(os.path.join(FORCING_DIR, solar_file))
            else:
                warnings.warn(
                            f'Requested solar data file "{solar_file}" not found!',
                        )
                raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!")
                       
        elif isolar == 4:# cmip5 mon-mean tsi in tim scale
            solar_file =  'solarconstant_cmip_mn.nc'
            if os.path.isfile(os.path.join(FORCING_DIR, solar_file)):
                data = xr.open_dataset(os.path.join(FORCING_DIR, solar_file))
            else:
                warnings.warn(
                            f'Requested solar data file "{solar_file}" not found!',
                        )    
                raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!")
                
        else:  
                warnings.warn("- !!! ERROR in selection of solar constant data",
                        f" source, ISOL = {isolar}",
                            )
                raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!")
    
    return solar_file, data

def sfc(FORCING_DIR):
    semis_file =  os.path.join(FORCING_DIR,"semisdata.nc") 
    data = xr.open_dataset(semis_file)
    return semis_file , data

def gases(FORCING_DIR, ictmflg):
    
    if ictmflg == 1:
        cfile1 = os.path.join(FORCING_DIR,'co2historicaldata_2016.nc') 
        var_names = ['iyr','cline','co2g1','co2g2','co2dat']
        if not os.path.isfile(cfile1):
            raise FileNotFoundError("   Can not find co2 data source file",
            "*** Stopped in subroutine gas_update !!",)
        
    #Opened CO2 climatology seasonal cycle data
    elif ictmflg == 2:
        cfile1 = os.path.join(FORCING_DIR,'') 
        var_names = ['cline','co2g1','co2g2','co2dat','gco2cyc']
        if not os.path.isfile(cfile1):
            raise FileNotFoundError("   Can not find co2 data source file",
            "*** Stopped in subroutine gas_update !!",)
    #  --- ...  read in co2 2-d data  
    ds = xr.open_dataset(cfile1)
    data_dict ={}
    for var in var_names:
        data_dict[var] = ds[var].values

    return data_dict



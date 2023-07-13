import xarray as xr
import glob
import numpy as np
import vcm
import vcm.catalog
from vcm.catalog import catalog
import xarray as xr
import netCDF4 as nc
import glob
import cftime
import datetime
import os

def main():
    path = '/home/paulah/data/era5/'
    variables = ['v_wind']
    setup_fregrid()
    for var in variables:
        check_time_steps_complete(path, var)
        weekdays = ['mon','tue','wed','thu','fri','sat','sun']
        for i, day in enumerate(weekdays):
            merge_into_file(path,var,i,day)
            print(day,' is merged.')
            regridded = regrid_to_cubed_sphere(path,var,day)
            print(day,' is regridded.')
            coarsened = coarsen(var,day)
            print(day,' is coarsened.')
            masked = mask(coarsened)
            print(day,' is masked.')
            masked.to_netcdf(path + var + '/masked/'+var+'_'+day+'.nc') 
            print(day,' is done.')
            
def setup_fregrid():
    #os.system("rm -rf fregrid-example")
    os.system("mkdir -p fregrid-example/source-data")
    os.system("mkdir -p fregrid-example/source-grid-data")
    os.system("gsutil -m cp gs://vcm-ml-intermediate/2023-06-19-fregrid-example-data/*.nc fregrid-example/source-data/")
    os.system("gsutil -m cp gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C96/*.nc fregrid-example/source-grid-data/")
    os.system("sudo docker run \
            -v $(pwd)/fregrid-example:/work \
            us.gcr.io/vcm-ml/post_process_run:latest \
            make_hgrid \
            --grid_type regular_lonlat_grid \
            --grid_name /work/era5_lonlat_grid \
            --nxbnds 2 \
            --nybnds 2 \
            --xbnds -180,180 \
            --ybnds -90,90 \
            --nlon 720\
            --nlat 362")
        
    os.system("sudo docker run \
        -v $(pwd)/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        make_solo_mosaic \
        --num_tiles 1\
        --dir /work \
        --tile_file era5_lonlat_grid.nc \
        --mosaic_name /work/era5_lonlat_grid_mosaic")
                
def check_time_steps_complete(path, var):    
    #check if data is complete (all time steps are there)
    time_step_list = []
    for f in glob.glob(path+ var +'/downloaded/*.nc'):
        dataset = xr.open_dataset(f)
        time_step_list.append(dataset['time'].data)
    dataset.close()
    arrs = list(np.concatenate(time_step_list))
    arrs.sort()
    for i,date in enumerate(arrs[1:]):
        if not (arrs[i]+np.timedelta64(1,'D')==date):
            print('missing day after:', arrs[i], i)
    print('all time steps checked.')
    
def merge_into_file(path,var,i,day):
    #create one file per weekday
    day_sublist = []
    for f in glob.glob(path+ var +'/downloaded/*.nc'):
        dataset = xr.open_dataset(f)
        for time in range(len(dataset.time)):
            data_at_day = dataset.isel(time=slice(time,time+1))
            if data_at_day.time.dt.weekday[0] == i:
                day_sublist.append(data_at_day)

        dataset.close()
    merged_data = xr.concat(day_sublist,dim='time')
    sorted_data = merged_data.sortby('time')
    sorted_data.to_netcdf(path + var + '/merged/' + var + '_' + day +'.nc')
    save_nc_int32_time(path + var + '/merged/'+var+'_'+day+'.nc', '/home/paulah/fregrid-example/'+var+'_'+day+'_i32_time.nc')
    #os.system("cp "+path+var+"/merged/"+var+"_"+day+"_i32_time.nc /home/paulah/fregrid-example")
    
def save_nc_int32_time(infile, outfile):
    in_nc = nc.Dataset(infile, "r")
    
    # load original time variable
    in_time = in_nc.variables["time"]
    in_time_values = in_time[:]
    as_dt = cftime.num2date(in_time_values, in_time.units, calendar=in_time.calendar)
    as_julian = cftime.date2num(as_dt, in_time.units, calendar="julian")
    in_nc.close()
    
    # save new file without time coordinate
    in_ds = xr.open_dataset(infile)
    in_ds.drop("time").to_netcdf(outfile)
    
    # add adjusted time dimension to the new file
    
    out_nc = nc.Dataset(outfile, "a")
    try:
        out_time = out_nc.createVariable("time", np.int32, ("time",))
        out_time[:] = as_julian
        for attr in in_time.ncattrs():
            if attr == "calendar":
                value = "julian"
            else:
                value = in_time.getncattr(attr)
            out_time.setncattr(attr, in_time.getncattr(attr))
    except:
        pass
    finally:
        out_nc.close()
        


def regrid_to_cubed_sphere(path,var,day):
    if var == 'temp2m':
        var_id = 't2m'
    elif var == 'u_wind':
        var_id = 'u10'
    elif var == 'v_wind':
        var_id = 'v10'
    else:
        var_id = var
    
    os.system("sudo docker run \
        -v $(pwd)/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        fregrid \
        --input_mosaic /work/era5_lonlat_grid_mosaic.nc \
        --output_mosaic /work/source-grid-data/grid_spec.nc \
        --input_file /work/"+var+"_"+day+"_i32_time.nc \
        --output_file /work/"+var+"_"+day+"_cubed.nc \
        --scalar_field "+var_id)
                
def coarsen(var,day):
    dt_lis = [xr.open_dataset(f,decode_times=False) for f in glob.glob('fregrid-example/'+var+'_'+day+'_cubed.*nc')]
    grid = vcm.catalog.catalog["grid/c96"].to_dask()
    new_lis = []
    for d in dt_lis:
        d = d.rename({"lon": "x", "lat": "y"}) 
        d['y'] = grid['y']
        d['x'] = grid['x']
        new_lis.append(d)
    ds = xr.concat(new_lis,dim='tile')
    if var == 'sst':
        cubed = ds.sst
    elif var == 'temp2m':
        cubed = ds.t2m
    elif var == 'u_wind':
        cubed = ds.u10
    elif var == 'v_wind':
        cubed = ds.v10
    #interpolate nans, because of mitmatch of era5 and c48 land-sea mask mismatch
    cubed = cubed.interpolate_na(dim='x')
    cubed = cubed.interpolate_na(dim='y')
    coarsened = vcm.cubedsphere.coarsen.weighted_block_average(cubed,grid['area'],2,x_dim='x',y_dim='y')
    return coarsened

def mask(coarsened):
    land_sea_mask_c48 = catalog['landseamask/c48'].read()
    masked = coarsened.where(land_sea_mask_c48['land_sea_mask']!=1)
    return masked
        

if __name__ == "__main__":
    main()
    

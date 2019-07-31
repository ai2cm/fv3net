import xarray as xr
import tqdm
import xesmf as xe
import fire
import numpy as np
from multiprocessing import Pool


def interp_h_plev_to_h_pfull_lat_lon(data):
    print('Hey, I am computing')
    lons = []
    for lon in data.lon:
        lats = []
        for lat in data.lat:
            h_plev = data.h_plev.sel(lat=lat, lon=lon).chunk({'plev': 31})
            pres = data.pres.sel(lat=lat, lon=lon)/100
            lats.append(h_plev.interp(plev=pres, method='cubic', kwargs={'fill_value':'extrapolate', 'bounds_error':False}))
        lons.append(xr.concat(lats, dim='lat'))
    return xr.concat(lons, dim='lon').transpose('pfull', 'lat', 'lon').rename('h_pfull')


def main(fn_in, fn_out):
    raw_data = xr.open_zarr(fn_in)
    pool = Pool(processes=4)
    times = [
        pool.apply_async(interp_h_plev_to_h_pfull_lat_lon, args=(raw_data.isel(time=itime),))
        for itime in range(len(raw_data.time))
    ]
    outputs = [p.get() for p in times]
    ds_out = xr.concat(outputs, dim='time')
#     ds_out = []
#     for itime in tqdm.tqdm(range(len(raw_data.time))):
#         ds_out.append(
#             interp_h_plev_to_h_pfull_lat_lon(raw_data.isel(time=itime))
#         )
#     ds_out = xr.concat(ds_out, dim='time')
    ds_out = xr.merge([ds_out, raw_data])
    ds_out.to_zarr(fn_out)

    
if __name__ == '__main__':
    fire.Fire(main)
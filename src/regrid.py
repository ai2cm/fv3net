import xarray as xr
import tqdm
import xesmf as xe
import fire
import numpy as np


def regrid_ds(fn_in, fn_out, factor):
    """
    A first very crude function to regrid FV3 data.
    """
    raw_data = xr.open_zarr(fn_in)
    raw_data = raw_data.rename({'grid_xt': 'lon', 'grid_yt': 'lat'})
    ddeg = raw_data.lon.diff('lon').values[0]
    ddeg_out = ddeg * factor
    
    ds_out = xr.Dataset({
        'lon': (['lon'], np.arange(ddeg_out/2, 360, ddeg_out)),
        'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out))
    })
    regridder = xe.Regridder(raw_data, ds_out, 'bilinear', reuse_weights=True)
    
    regridded_das = []
    for var in tqdm.tqdm(raw_data):
        da = raw_data[var]
        if 'lon' in da.coords and 'lat' in da.coords:
            regridded_das.append(regridder(da))
    
    ds_out = xr.Dataset({da.name: da for da in regridded_das})
    ds_out.to_zarr(fn_out)
    
    
if __name__ == '__main__':
    fire.Fire(regrid_ds)
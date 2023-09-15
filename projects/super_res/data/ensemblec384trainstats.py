import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

precip_folder = Path('./ensemble_c384_trainstats')
precip_folder.mkdir(exist_ok = True, parents = True)

ENSEMBLE = 10

channels = ["PRATEsfc"]
chl = {}

for channel in channels:
    
    chl[channel] = {}
    chl[channel]['min'] = np.PINF
    chl[channel]['max'] = np.NINF

for member in tqdm(range(1, ENSEMBLE + 1)):
    
    c384 = xr.open_zarr(f"/extra/ucibdl0/shared/data/fv3gfs/c384_precip_ave/{member:04d}/sfc_8xdaily_ave.zarr")
    
    for channel in tqdm(channels):
        
        channel_384 = c384[channel]

        for idx in tqdm(range(397)):
            
            channel_384_slice = channel_384.isel(time = slice(idx*8, (idx+1)*8))
            channel_384_max = channel_384_slice.max().values
            channel_384_min = channel_384_slice.min().values

            if channel_384_min < chl[channel]['min']:
                
                chl[channel]['min'] = channel_384_min
            
            if channel_384_max > chl[channel]['max']:
                
                chl[channel]['max'] = channel_384_max

# save the chl dictionary as pickle
with open(precip_folder / 'chl.pkl', 'wb') as f:
    pickle.dump(chl, f)
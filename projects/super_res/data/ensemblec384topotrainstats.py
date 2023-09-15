import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

precip_folder = Path('./ensemble_c384_trainstats')
precip_folder.mkdir(exist_ok = True, parents = True)

ENSEMBLE = 10

channels = ["zsurf"]
chl = {}

for channel in channels:
    
    chl[channel] = {}
    chl[channel]['min'] = np.PINF
    chl[channel]['max'] = np.NINF

for member in tqdm(range(1, ENSEMBLE + 1)):
    
    topo = xr.open_zarr(f"/extra/ucibdl0/shared/data/fv3gfs/c384_topo/{member:04d}/atmos_static.zarr")
    
    for channel in tqdm(channels):
        channel_384 = topo[channel]
        channel_384_min = channel_384.min().values
        channel_384_max = channel_384.max().values
        if channel_384_min < chl[channel]['min']:
            chl[channel]['min'] = channel_384_min
        if channel_384_max > chl[channel]['max']:
            chl[channel]['max'] = channel_384_max

# save the chl dictionary as pickle
with open(precip_folder / 'topo.pkl', 'wb') as f:
    pickle.dump(chl, f)
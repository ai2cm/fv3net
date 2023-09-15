import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

precip_folder = Path('./ensemble_c48_trainstats')
precip_folder.mkdir(exist_ok = True, parents = True)

ENSEMBLE = 10

channels = ["PRATEsfc_coarse", "UGRD10m_coarse", "VGRD10m_coarse", "TMPsfc_coarse", "CPRATsfc_coarse", "DSWRFtoa_coarse"]
atm_channels = ["ps_coarse", "u700_coarse", "v700_coarse", "vertically_integrated_liq_wat_coarse", "vertically_integrated_sphum_coarse"]

chl, atm_chl = {}, {}

for channel in channels:
    
    chl[channel] = {}
    chl[channel]['min'] = np.PINF
    chl[channel]['max'] = np.NINF

for channel in atm_channels:

    atm_chl[channel] = {}
    atm_chl[channel]['min'] = np.PINF
    atm_chl[channel]['max'] = np.NINF

for member in tqdm(range(1, ENSEMBLE + 1)):
    
    c48 = xr.open_zarr(f"/extra/ucibdl0/shared/data/fv3gfs/c48_precip_plus_more_ave/{member:04d}/sfc_8xdaily_ave_coarse.zarr")
    c48_atm = xr.open_zarr(f"/extra/ucibdl0/shared/data/fv3gfs/c48_atmos_ave/{member:04d}/atmos_8xdaily_ave_coarse.zarr")
    
    for channel in tqdm(channels):
        channel_48 = c48[channel]
        channel_48_min = channel_48.min().values
        channel_48_max = channel_48.max().values
        if channel_48_min < chl[channel]['min']:
            chl[channel]['min'] = channel_48_min
        if channel_48_max > chl[channel]['max']:
            chl[channel]['max'] = channel_48_max

    for channel in tqdm(atm_channels):
        channel_48 = c48_atm[channel]
        channel_48_min = channel_48.min().values
        channel_48_max = channel_48.max().values
        if channel_48_min < atm_chl[channel]['min']:
            atm_chl[channel]['min'] = channel_48_min
        if channel_48_max > atm_chl[channel]['max']:
            atm_chl[channel]['max'] = channel_48_max

# save the chl dictionary as pickle
with open(precip_folder / 'chl.pkl', 'wb') as f:
    pickle.dump(chl, f)

# save the atm_chl dictionary as pickle
with open(precip_folder / 'atm_chl.pkl', 'wb') as f:
    pickle.dump(atm_chl, f)
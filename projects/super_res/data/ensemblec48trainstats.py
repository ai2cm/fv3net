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
    
    c48 = xr.open_zarr(f"/data/prakhars/ensemble/c48_precip_plus_more_ave/{member:04d}/sfc_8xdaily_ave_coarse.zarr")
    c48_atm = xr.open_zarr(f"/data/prakhars/ensemble/c48_atmos_ave/{member:04d}/atmos_8xdaily_ave_coarse.zarr")
    
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


    

# channels = ["PRATEsfc_coarse"]
# c384_np = np.stack([c384[channel].values for channel in channels], axis = 2)
# c48_np = np.stack([c48[channel].values for channel in channels], axis = 2)

# np.save('only_precip/c384_gmin.npy', c384_np.min())
# np.save('only_precip/c48_gmin.npy', c48_np.min())

# # calculate split (80/20)
# split = int(c384_np.shape[1] * 0.8)

# # compute statistics on training set
# c384_min, c384_max, c48_min, c48_max = c384_np[:, :split, :, :, :].min(), c384_np[:, :split, :, :, :].max(), c48_np[:, :split, :, :, :].min(), c48_np[:, :split, :, :, :].max() 

# # normalize
# c384_norm= (c384_np - c384_min) / (c384_max - c384_min)
# c48_norm = (c48_np - c48_min) / (c48_max - c48_min)

# np.save('only_precip/c384_min.npy', c384_min)
# np.save('only_precip/c384_max.npy', c384_max)
# np.save('only_precip/c48_min.npy', c48_min)
# np.save('only_precip/c48_max.npy', c48_max)
# np.save('only_precip/c48_norm.npy', c48_norm)
# np.save('only_precip/c384_norm.npy', c384_norm)

# c384_lnp = np.log(c384_np - c384_np.min() + 1e-14)
# c48_lnp = np.log(c48_np - c48_np.min() + 1e-14)

# # compute statistics on training set
# c384_lmin, c384_lmax, c48_lmin, c48_lmax = c384_lnp[:, :split, :, :, :].min(), c384_lnp[:, :split, :, :, :].max(), c48_lnp[:, :split, :, :, :].min(), c48_lnp[:, :split, :, :, :].max() 

# # normalize
# c384_lnorm= (c384_lnp - c384_lmin) / (c384_lmax - c384_lmin)
# c48_lnorm = (c48_lnp - c48_lmin) / (c48_lmax - c48_lmin)

# np.save('only_precip/c384_lgmin.npy', c384_lmin)
# np.save('only_precip/c384_lgmax.npy', c384_lmax)
# np.save('only_precip/c48_lgmin.npy', c48_lmin)
# np.save('only_precip/c48_lgmax.npy', c48_lmax)
# np.save('only_precip/c48_lgnorm.npy', c48_lnorm)
# np.save('only_precip/c384_lgnorm.npy', c384_lnorm)
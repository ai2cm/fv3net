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
    
    c384 = xr.open_zarr(f"/data/prakhars/ensemble/c384_precip_ave/{member:04d}/sfc_8xdaily_ave.zarr")
    
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
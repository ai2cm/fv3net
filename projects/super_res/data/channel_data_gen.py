import xarray as xr
import numpy as np
from pathlib import Path

channel_folder = Path('./more_channels')
channel_folder.mkdir(exist_ok = True, parents = True)

c384 = xr.open_zarr("gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt_coarse": "x", "grid_yt_coarse": "y"})
c48 = xr.open_zarr("gs://vcm-ml-intermediate/2021-10-12-PIRE-c48-post-spinup-verification/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt": "x", "grid_yt": "y"})

channels = ["UGRD10m_coarse", "VGRD10m_coarse", "tsfc_coarse", "CPRATEsfc_coarse"]
c384_np = np.stack([c384[channel].values for channel in channels], axis = 2)
c48_np = np.stack([c48[channel].values for channel in channels], axis = 2)

split = int(c384_np.shape[1] * 0.8)

# compute statistics on training set
c384_min, c384_max, c48_min, c48_max = c384_np[:, :split, :, :, :].min(axis=(0,1,3,4)).reshape(1,1,4,1,1), c384_np[:, :split, :, :, :].max(axis=(0,1,3,4)).reshape(1,1,4,1,1), c48_np[:, :split, :, :, :].min(axis=(0,1,3,4)).reshape(1,1,4,1,1), c48_np[:, :split, :, :, :].max(axis=(0,1,3,4)).reshape(1,1,4,1,1) 

# normalize
c384_norm= (c384_np - c384_min) / (c384_max - c384_min)
c48_norm = (c48_np - c48_min) / (c48_max - c48_min)

np.save('c384_min.npy', c384_min)
np.save('c384_max.npy', c384_max)
np.save('c48_min.npy', c48_min)
np.save('c48_max.npy', c48_max)
np.save('c48_norm.npy', c48_norm)
np.save('c384_norm.npy', c384_norm)
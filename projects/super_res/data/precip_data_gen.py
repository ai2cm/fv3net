import xarray as xr
import numpy as np
from pathlib import Path

precip_folder = Path('./only_precip')
precip_folder.mkdir(exist_ok = True, parents = True)

c384 = xr.open_zarr("gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt_coarse": "x", "grid_yt_coarse": "y"})
c48 = xr.open_zarr("gs://vcm-ml-intermediate/2021-10-12-PIRE-c48-post-spinup-verification/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt": "x", "grid_yt": "y"})

channels = ["PRATEsfc_coarse"]
c384_np = np.stack([c384[channel].values for channel in channels], axis = 2)
c48_np = np.stack([c48[channel].values for channel in channels], axis = 2)

np.save('only_precip/c384_gmin.npy', c384_np.min())
np.save('only_precip/c48_gmin.npy', c48_np.min())

# calculate split (80/20)
split = int(c384_np.shape[1] * 0.8)

# compute statistics on training set
c384_min, c384_max, c48_min, c48_max = c384_np[:, :split, :, :, :].min(), c384_np[:, :split, :, :, :].max(), c48_np[:, :split, :, :, :].min(), c48_np[:, :split, :, :, :].max() 

# normalize
c384_norm= (c384_np - c384_min) / (c384_max - c384_min)
c48_norm = (c48_np - c48_min) / (c48_max - c48_min)

np.save('only_precip/c384_min.npy', c384_min)
np.save('only_precip/c384_max.npy', c384_max)
np.save('only_precip/c48_min.npy', c48_min)
np.save('only_precip/c48_max.npy', c48_max)
np.save('only_precip/c48_norm.npy', c48_norm)
np.save('only_precip/c384_norm.npy', c384_norm)

c384_lnp = np.log(c384_np - c384_np.min() + 1e-14)
c48_lnp = np.log(c48_np - c48_np.min() + 1e-14)

# compute statistics on training set
c384_lmin, c384_lmax, c48_lmin, c48_lmax = c384_lnp[:, :split, :, :, :].min(), c384_lnp[:, :split, :, :, :].max(), c48_lnp[:, :split, :, :, :].min(), c48_lnp[:, :split, :, :, :].max() 

# normalize
c384_lnorm= (c384_lnp - c384_lmin) / (c384_lmax - c384_lmin)
c48_lnorm = (c48_lnp - c48_lmin) / (c48_lmax - c48_lmin)

np.save('only_precip/c384_lgmin.npy', c384_lmin)
np.save('only_precip/c384_lgmax.npy', c384_lmax)
np.save('only_precip/c48_lgmin.npy', c48_lmin)
np.save('only_precip/c48_lgmax.npy', c48_lmax)
np.save('only_precip/c48_lgnorm.npy', c48_lnorm)
np.save('only_precip/c384_lgnorm.npy', c384_lnorm)
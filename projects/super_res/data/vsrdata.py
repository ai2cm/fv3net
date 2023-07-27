import xarray as xr
import numpy as np
from torch.utils.data import Dataset

class VSRDataset(Dataset):
    
    def __init__(self, channels, mode, length, logscale = False, quick = True):
        '''
        Args:
            channels (list): list of channels to use
            mode (str): train or val
            length (int): length of sequence
            logscale (bool): whether to logscale the data
            quick (bool): whether to load data from bucket or from local (local only supports single precipitation channel)
        '''
        
        # expected sequence length
        self.length = length

        # mode
        self.mode = mode

        if not quick:
            # load data from bucket
            # shape : (tile, time, y, x)
            c384 = xr.open_zarr("gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt_coarse": "x", "grid_yt_coarse": "y"})
            c48 = xr.open_zarr("gs://vcm-ml-intermediate/2021-10-12-PIRE-c48-post-spinup-verification/pire_atmos_phys_3h_coarse.zarr").rename({"grid_xt": "x", "grid_yt": "y"})
            
            # convert to numpy
            # shape : (tile, time, channel, y, x)
            c384_np = np.stack([c384[channel].values for channel in channels], axis = 2)
            c48_np = np.stack([c48[channel].values for channel in channels], axis = 2)

            if logscale:
                c384_np = np.log(c384_np - c384_np.min() + 1e-14)
                c48_np = np.log(c48_np - c48_np.min() + 1e-14)

            # calculate split (80/20)
            split = int(c384_np.shape[1] * 0.8)

            # compute statistics on training set
            c384_min, c384_max, c48_min, c48_max = c384_np[:, :split, :, :, :].min(), c384_np[:, :split, :, :, :].max(), c48_np[:, :split, :, :, :].min(), c48_np[:, :split, :, :, :].max() 

            # normalize
            c384_norm= (c384_np - c384_min) / (c384_max - c384_min)
            c48_norm = (c48_np - c48_min) / (c48_max - c48_min)

            if mode == 'train':
                
                self.X = c48_norm[:, :split, :, :, :]
                self.y = c384_norm[:, :split, :, :, :]
                
            elif mode == 'val':
                
                self.X = c48_norm[:, split:, :, :, :]
                self.y = c384_norm[:, split:, :, :, :]

        else:

            c384_norm= np.load("data/only_precip/c384_norm.npy")
            c48_norm = np.load("data/only_precip/c48_norm.npy")

            # calculate split (80/20)
            split = int(c384_norm.shape[1] * 0.8)

            if mode == 'train':
                
                self.X = c48_norm[:, :split, :, :, :]
                self.y = c384_norm[:, :split, :, :, :]
                
            elif mode == 'val':
                
                self.X = c48_norm[:, split:, :, :, :]
                self.y = c384_norm[:, split:, :, :, :]

    def __len__(self):
        
        return self.X.shape[1] - self.length + 1

    def __getitem__(self, idx):
        
        # load a random tile index

        if self.mode == 'train':
            tile = np.random.randint(0, self.X.shape[0])
        
        elif self.mode == 'val':
            tile = 0

        lowres = self.X[tile, idx:idx+self.length, :, :, :]
        highres = self.y[tile, idx:idx+self.length, :, :, :]

        return {'LR' : lowres, 'HR' : highres}
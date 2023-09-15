import numpy as np
import xarray as xr
from torch.utils.data import Dataset

class VSRDataset(Dataset):
    
    def __init__(self, mode, length, logscale = False, multi = False):
        '''
        Args:
            channels (list): list of channels to use
            mode (str): train or val
            length (int): length of sequence
            logscale (bool): whether to logscale the data
            quick (bool): whether to load data from bucket or from local (local only supports single precipitation channel)
        '''

        # load data
        self.y = xr.open_zarr("/data/prakhars/pire_atmos_phys_3h_c384.zarr")
        self.X = xr.open_zarr('/data/prakhars/pire_atmos_phys_3h_c48.zarr')
        
        # expected sequence length
        self.length = length

        # mode
        self.mode = mode

        self.logscale = logscale

        if logscale:

            self.c384_gmin = np.load('data/only_precip/c384_gmin.npy')
            self.c48_gmin = np.load('data/only_precip/c48_gmin.npy')
            self.c384_lgmin = np.load('data/only_precip/c384_lgmin.npy')
            self.c384_lgmax = np.load('data/only_precip/c384_lgmax.npy')
            self.c48_lgmin = np.load('data/only_precip/c48_lgmin.npy')
            self.c48_lgmax = np.load('data/only_precip/c48_lgmax.npy')

        else:

            self.c384_min = np.load('data/only_precip/c384_min.npy')
            self.c384_max = np.load('data/only_precip/c384_max.npy')
            self.c48_min = np.load('data/only_precip/c48_min.npy')
            self.c48_max = np.load('data/only_precip/c48_max.npy')
            
        self.time_steps = self.X.time.shape[0]
        self.tiles = self.X.tile.shape[0]

        self.multi = multi

        if multi:

            self.channels = ["PRATEsfc_coarse", "UGRD10m_coarse", "VGRD10m_coarse", "tsfc_coarse", "CPRATEsfc_coarse"]
            self.topo384 = np.load("data/topography/topo384_norm.npy")
            self.c384_multimin = np.load('data/more_channels/c384_min.npy')
            self.c384_multimax = np.load('data/more_channels/c384_max.npy')
            self.c48_multimin = np.load('data/more_channels/c48_min.npy')
            self.c48_multimax = np.load('data/more_channels/c48_max.npy')
        
        else:

            self.channels = ["PRATEsfc_coarse"]

        if mode == 'train':
            
            self.indices = list(range(int(self.time_steps * 0.8) - self.length + 1))
            
        elif mode == 'val':
            
            self.indices = list(range(int(self.time_steps * 0.8), self.time_steps - self.length + 1))

    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        time_idx = self.indices[idx]
        if self.mode == 'train':
            tile = idx % self.tiles
        else:
            tile = 0
        
        lowres = self.X.isel(time = slice(time_idx, time_idx + self.length), tile = tile)
        lowres = np.stack([lowres[channel].values for channel in self.channels], axis = 1)
        highres = self.y.isel(time = slice(time_idx, time_idx + self.length), tile = tile)
        highres = np.stack([highres[channel].values for channel in self.channels[0:1]], axis = 1)
        
        if self.logscale:

            lowres[:,0:1,:,:] = np.log(lowres[:,0:1,:,:] - self.c48_gmin + 1e-14)
            highres = np.log(highres - self.c384_gmin + 1e-14)
            lowres[:,0:1,:,:] = (lowres[:,0:1,:,:] - self.c48_lgmin) / (self.c48_lgmax - self.c48_lgmin)
            highres = (highres - self.c384_lgmin) / (self.c384_lgmax - self.c384_lgmin)

        else:

            lowres[:,0:1,:,:] = (lowres[:,0:1,:,:] - self.c48_min) / (self.c48_max - self.c48_min)
            highres = (highres - self.c384_min) / (self.c384_max - self.c384_min)

        if self.multi:
            
            lowres[:,1:,:,:] = (lowres[:,1:,:,:] - self.c48_multimin) / (self.c48_multimax - self.c48_multimin)
            topo = self.topo384[tile,:,:]
            topo = np.repeat(topo.reshape((1,1,384,384)), self.length, axis = 0)
            highres = np.concatenate((highres, topo), axis = 1)
            
        return {'LR' : lowres, 'HR' : highres}
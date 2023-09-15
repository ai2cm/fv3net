import torch
import pickle
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
            multi (bool): whether to use multi-channel data
        '''

        ENSEMBLE = 11

        # load data
        self.X, self.X_, self.y, self.topo = {}, {}, {}, {}

        PATH = "/extra/ucibdl0/shared/data/fv3gfs"

        for member in range(1, ENSEMBLE + 1):

            self.X[member] = xr.open_zarr(f"{PATH}/c48_precip_plus_more_ave/{member:04d}/sfc_8xdaily_ave_coarse.zarr")
            self.X_[member] = xr.open_zarr(f"{PATH}/c48_atmos_ave/{member:04d}/atmos_8xdaily_ave_coarse.zarr")
            self.y[member] = xr.open_zarr(f"{PATH}/c384_precip_ave/{member:04d}/sfc_8xdaily_ave.zarr")
            self.topo[member] = xr.open_zarr(f"{PATH}/c384_topo/{member:04d}/atmos_static.zarr")
        
        # expected sequence length
        self.length = length

        self.mode = mode
        self.logscale = logscale
        self.multi = multi

        self.time_steps = self.X[1].time.shape[0]
        self.tiles = self.X[1].tile.shape[0]

        # load statistics
        with open("data/ensemble_c48_trainstats/chl.pkl", 'rb') as f:

            self.c48_chl = pickle.load(f)
        
        with open("data/ensemble_c48_trainstats/atm_chl.pkl", 'rb') as f:
            
            self.c48_atm_chl = pickle.load(f)

        with open("data/ensemble_c48_trainstats/log_chl.pkl", 'rb') as f:
            
            self.c48_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:

            self.c384_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:

            self.c384_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/topo.pkl", 'rb') as f:

            self.c384_topo = pickle.load(f)

        if multi:

            self.c48_channels = ["PRATEsfc_coarse", "UGRD10m_coarse", "VGRD10m_coarse", "TMPsfc_coarse", "CPRATsfc_coarse", "DSWRFtoa_coarse"]
            self.c48_channels_atmos = ["ps_coarse", "u700_coarse", "v700_coarse", "vertically_integrated_liq_wat_coarse", "vertically_integrated_sphum_coarse"]
            self.c384_channels = ["PRATEsfc"]

        else:

            self.c48_channels = ["PRATEsfc_coarse"]
            self.c384_channels = ["PRATEsfc"]

        self.indices = list(range(self.time_steps - self.length + 1))

    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        time_idx = self.indices[idx]

        if self.mode == 'train':
            
            np.random.seed()
            tile = np.random.randint(self.tiles)
            member = np.random.randint(10) + 1
        
        else:
            
            tile = idx % self.tiles
            member = 11

        X = self.X[member].isel(time = slice(time_idx, time_idx + self.length), tile = tile)
        X_ = self.X_[member].isel(time = slice(time_idx, time_idx + self.length), tile = tile)
        y = self.y[member].isel(time = slice(time_idx, time_idx + self.length), tile = tile)

        if self.multi:

            X = np.stack([X[channel].values for channel in self.c48_channels], axis = 1)
            X_ = np.stack([X_[channel].values for channel in self.c48_channels_atmos], axis = 1)
            y = np.stack([y[channel].values for channel in self.c384_channels], axis = 1)
            topo = self.topo[member].isel(tile = tile)
            topo = topo['zsurf'].values
            topo = np.repeat(topo.reshape((1,1,384,384)), self.length, axis = 0)

        else:

            X = np.stack([X[channel].values for channel in self.c48_channels], axis = 1)
            y = np.stack([y[channel].values for channel in self.c384_channels], axis = 1)

        
        if self.logscale:

            X[:,0:1,:,:] = np.log(X[:,0:1,:,:] - self.c48_chl["PRATEsfc_coarse"]['min'] + 1e-14)
            y = np.log(y - self.c384_chl["PRATEsfc"]['min'] + 1e-14)
            X[:,0:1,:,:] = (X[:,0:1,:,:] - self.c48_log_chl["PRATEsfc_coarse"]['min']) / (self.c48_log_chl["PRATEsfc_coarse"]['max'] - self.c48_log_chl["PRATEsfc_coarse"]['min'])
            y = (y - self.c384_log_chl["PRATEsfc"]['min']) / (self.c384_log_chl["PRATEsfc"]['max'] - self.c384_log_chl["PRATEsfc"]['min'])

        else:

            X[:,0:1,:,:] = (X[:,0:1,:,:] - self.c48_chl["PRATEsfc_coarse"]['min']) / (self.c48_chl["PRATEsfc_coarse"]['max'] - self.c48_chl["PRATEsfc_coarse"]['min'])
            y = (y - self.c384_chl["PRATEsfc"]['min']) / (self.c384_chl["PRATEsfc"]['max'] - self.c384_chl["PRATEsfc"]['min'])

        if self.multi:

            for i in range(1, X.shape[1]):

                X[:,i,:,:] = (X[:,i,:,:] - self.c48_chl[self.c48_channels[i]]['min']) / (self.c48_chl[self.c48_channels[i]]['max'] - self.c48_chl[self.c48_channels[i]]['min'])

            for i in range(X_.shape[1]):

                X_[:,i,:,:] = (X_[:,i,:,:] - self.c48_atm_chl[self.c48_channels_atmos[i]]['min']) / (self.c48_atm_chl[self.c48_channels_atmos[i]]['max'] - self.c48_atm_chl[self.c48_channels_atmos[i]]['min'])

            topo = (topo - self.c384_topo["zsurf"]['min']) / (self.c384_topo["zsurf"]['max'] - self.c384_topo["zsurf"]['min'])

            X = np.concatenate((X, X_), axis = 1)
            y = np.concatenate((y, topo), axis = 1)

        return {'LR' : X, 'HR' : y}
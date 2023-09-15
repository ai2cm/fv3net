import numpy as np
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
        
        # expected sequence length
        self.length = length

        # mode
        self.mode = mode

        # data shape : (num_tiles, num_frames, num_channels, height, width)
        # num_tiles = 6; num_frames = 2920, num_channels = 1
        if logscale:

            c384_norm= np.load("data/only_precip/c384_lgnorm.npy")
            c48_norm = np.load("data/only_precip/c48_lgnorm.npy")

        else:

            c384_norm= np.load("data/only_precip/c384_norm.npy")
            c48_norm = np.load("data/only_precip/c48_norm.npy")
        
        t, f, c, h, w = c384_norm.shape

        if multi:

            # load more channels, order : ("UGRD10m_coarse", "VGRD10m_coarse", "tsfc_coarse", "CPRATEsfc_coarse")
            c48_norm_more = np.load("data/more_channels/c48_norm.npy")
            c48_norm = np.concatenate((c48_norm, c48_norm_more), axis = 2)

            # load topography, shape : (num_tiles, height, width)
            # reshaping to match data shape
            topo384 = np.repeat(np.load("data/topography/topo384_norm.npy").reshape((t, 1, c, 384, 384)), f, axis = 1)
            c384_norm = np.concatenate((c384_norm, topo384), axis = 2)

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

        # tensor shape : (length, num_channels, height, width)
        lowres = self.X[tile, idx:idx+self.length, :, :, :]
        highres = self.y[tile, idx:idx+self.length, :, :, :]

        return {'LR' : lowres, 'HR' : highres}
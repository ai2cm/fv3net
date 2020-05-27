from fv3net.regression import loaders
import xarray as xr
import logging
from typing import Sequence


logging.getLogger(__name__)


def time_average(batches: Sequence[xr.Dataset], time_dim = 'initial_time') -> xr.Dataset:
    '''Average over time dimension'''
    
    ds = xr.concat(batches, dim=time_dim)
    
    return ds.mean(dim=time_dim, keep_attrs=True)

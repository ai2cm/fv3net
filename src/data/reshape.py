"""Convert netCDF dataset to a memory efficient layout for machine learning

The original dataset should have the dimensions ('time', 'z', 'y', 'x'). This
script combines the y and x dimensions into a sample dimension with a specified
chunk size. This sample dimension is optionally shufffled, and then saved to a
zarr archive.

The saved archive is suitable to use with uwnet.train
"""
import xarray as xr
import numpy as np
import argparse
from src.data import open_data

output_file = "data/processed/shuffled.zarr"
shuffle = True
ds = open_data(sources=True)

variables = 'u v w temp q1 q2 qv pres z'.split()
sample_dims = ['grid_yt', 'grid_xt']

# stack data
variables = list(variables) # needs to be a list for xarray
stacked = (ds[variables]
           .stack(sample=sample_dims)
           .drop('sample'))

# shuffle samples
if shuffle:
    n = len(stacked.sample)
    indices = np.random.choice(n, n, replace=False)
    stacked = stacked.isel(sample=indices)

chunked = stacked.chunk({'sample': chunk_size})

# save to disk
chunked.to_zarr(output_file)

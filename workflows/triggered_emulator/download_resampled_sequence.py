#!/usr/bin/env python
# coding: utf-8
import loaders
import numpy as np
import trigger
import xarray as xr

# In[ ]:


url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free"


mapper = loaders.mappers.open_baseline_emulator(url)

input_variables = [
    "air_temperature",
    "specific_humidity",
    "cos_zenith_angle",
    "surface_geopotential",
]
output_variables = ["dQ1", "dQ2"]

sequence = loaders.batches.batches_from_mapper(
    mapper, variable_names=input_variables + output_variables, timesteps_per_batch=5,
)

resampled_seq = sequence.map(trigger.resample_inactive)

local = resampled_seq.local("output")

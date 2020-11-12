#!/usr/bin/env python
# coding: utf-8
import os
import sys
import loaders
import numpy as np
from functools import partial
import trigger
import xarray as xr
import loaders.mappers

import yaml

output = sys.argv[1]


def is_active(ds: xr.Dataset):
    return ds.surface_precipitation_rate > (5 / 86400)


if os.path.exists(output):
    sys.exit(0)


with open("hybrid-mapper.yaml") as f:
    config = yaml.safe_load(f)

mapper = getattr(loaders.mappers, config["mapping_function"])(
    None, **config["mapping_kwargs"]
)

input_variables = [
    "air_temperature",
    "specific_humidity",
    "cos_zenith_angle",
    "surface_geopotential",
    "dQ1",
    "dQ2",
    "surface_precipitation_rate"
]

sequence = loaders.batches.batches_from_mapper(
    mapper, variable_names=input_variables, timesteps_per_batch=5,
)
resampled_seq = sequence.map(partial(trigger.resample_inactive, is_active=is_active))
local = resampled_seq.local(sys.argv[1])

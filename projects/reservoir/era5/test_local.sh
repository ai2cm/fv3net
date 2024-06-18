#!/bin/bash

python3 regrid_latlon_era5.py \
    gs://vcm-ml-scratch/andrep/reservoir/era5/era5_daily_all_360x180_v2.zarr \
    gs://vcm-ml-scratch/andrep/era5_regrid/test_c48_regrid.zarr \
    --template_path gs://vcm-ml-scratch/andrep/reservoir/era5/c48_template.nc \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/andrep/temp/ \
    --experiments use_runner_v2 \
    --runner DirectRunner \
    --test

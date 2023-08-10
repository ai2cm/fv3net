#!/bin/bash

python daily_avg.py \
    gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr \
    --output_path gs://vcm-ml-scratch/andrep/era5_regrid/test_regridded.zarr \
    --daily_template_path single_day_template.nc \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/andrep/temp/ \
    --experiments use_runner_v2 \
    --runner DirectRunner \
    --test

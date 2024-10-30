#!/bin/bash

python3 regrid_latlon_era5.py \
    gs://vcm-ml-intermediate/reservoir/era5/era5_daily_all_360x180_v2_filled_na.zarr/ \
    gs://vcm-ml-intermediate/reservoir/era5_regrid/era5_daily_all_c48_regrid_v2.zarr \
    --template_path gs://vcm-ml-scratch/andrep/reservoir/era5/c48_template.nc \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/andrep/temp/ \
    --experiments use_runner_v2 \
    --runner DataflowRunner \
    --sdk_container_image gcr.io/vcm-ml/dataflow-xbeam:v3-oct2023 \
    --sdk_location container \
    --num_workers 4 \
    --machine_type n1-standard-4

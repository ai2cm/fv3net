#!/bin/bash

# Deploy on gcp
python daily_avg.py \
    gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr \
    --output_path gs://vcm-ml-scratch/andrep/era5_regrid/daily_avg.zarr \
    --daily_template single_day_template.nc \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/andrep/temp/ \
    --experiments use_runner_v2 \
    --runner DataflowRunner \
    --sdk_container_image gcr.io/vcm-ml/dataflow-xbeam:v2 \
    --sdk_location container \
    --num_workers 4 \
    --machine_type n1-standard-4

#!/bin/bash


python -m save_ranks \
    gs://vcm-ml-experiments/spencerc/2022-01-19/n2f-25km-unperturbed-snoalb/fv3gfs_run/state_after_timestep.zarr \
    gs://vcm-ml-scratch/annak/2023-02-27/rank_data/ \
    2 \
    2 \
    --stop-time 20180815.000000 \
    --variables air_temperature specific_humidity \
    --time-chunks 40


python -m save_ranks \
    gs://vcm-ml-experiments/spencerc/2022-01-19/n2f-25km-unperturbed-snoalb/fv3gfs_run/state_after_timestep.zarr \
    gs://vcm-ml-experiments/reservoir-computing-offline/data/n2f-25km/train/start_20190215_end_20190615 \
    2 \
    2 \
    --start-time 20190215.000000 \
    --stop-time 20190615.000000 \
    --variables air_temperature specific_humidity \
    --time-chunks 40 \
    --ranks 0 1


python -m save_ranks \
    gs://vcm-ml-experiments/spencerc/2022-01-19/n2f-25km-unperturbed-snoalb/fv3gfs_run/state_after_timestep.zarr \
    gs://vcm-ml-experiments/reservoir-computing-offline/data/n2f-25km/val/start_20190615_end_2019_0715 \
    2 \
    2 \
    --start-time 20190615.000000 \
    --stop-time 20190715.000000 \
    --variables air_temperature specific_humidity \
    --time-chunks 40 \
    --ranks 0 1

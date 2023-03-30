#!/bin/bash

python -m create_offline_report \
    final_offline_eval/offline_ML_Tquv_seed_3.yaml \
    --n_weeks 16 \
    --time_sample_freq 12H \
    --variables air_temperature specific_humidity
#!/bin/bash

python -m fv3fit.train \
    /home/AnnaK/fv3net/projects/reservoir/fv3/train_config.yaml \
    /home/AnnaK/fv3net/projects/reservoir/fv3/train_data.yaml \
    gs://vcm-ml-scratch/annak/2023-04-19/persistence_rc_no_encoder_T \
    --validation-data-config /home/AnnaK/fv3net/projects/reservoir/fv3/train_data.yaml \
    --no-wandb
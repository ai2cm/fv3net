#!/bin/bash

python -m train_ks \
    configs/ks_config.yaml \
    configs/train_config.yaml \
    gs://vcm-ml-scratch/annak/2022-10-17/rc-model-1d-ks

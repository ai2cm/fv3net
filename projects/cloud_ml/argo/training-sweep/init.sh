#!/bin/bash

sweep_config=$1
SWEEP_ID=$(wandb sweep --entity ai2cm --project radiation-cloud-ml ${sweep_config}-sweep.yaml 2>&1 | awk '/ID:/{print $6}')
echo "export SWEEP_ID=$SWEEP_ID"
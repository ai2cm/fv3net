#!/bin/bash

cat << EOF  > config.yaml
base_version: v0.5
namelist:
  coupler_nml:
    days: 1 # total length
    hours: 0
    minutes: 0
    seconds: 0
    dt_atmos: 900 # seconds
    dt_ocean: 900
    restart_secs: 0 # seconds - frequency to save restarts
  atmos_model_nml:
    fhout: 0.25 # hours - frequency to save physics outputs
  gfs_physics_nml:
    fhzero: 0.25 # hours - frequency at which precip is set back to zero
  fv_core_nml:
    n_split: 6 # num dynamics steps per physics step
EOF

ROOT=/data/prognostic-runs
RUN=$ROOT/$(date -Iseconds)
echo "Running in $RUN"

mkdir -p $ROOT

python3 prepare_config.py config.yaml gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts 20160805.000000 > fv3config.yaml
runfv3 run-native fv3config.yaml "$RUN" sklearn_runfile.py
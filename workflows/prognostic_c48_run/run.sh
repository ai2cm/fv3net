#!/bin/bash

# from https://github.com/VulcanClimateModeling/vcm-workflow-control/blob/86a4d116d86d9e06c76abf81ac18d33915a3970d/2021-05-28-online-training/emulator-prognostic.yaml
cat << EOF  > config.yaml
base_version: v0.5
forcing: gs://vcm-fv3config/data/base_forcing/v1.1/
online_emulator:
  checkpoint: ai2cm/emulator-noah/model:v9
  train: false
  online: true
namelist:
  coupler_nml:
    days: 0 # total length
    hours: 12
    minutes: 0
    seconds: 0
    dt_atmos: 900 # seconds
    dt_ocean: 900
    restart_secs: 0 # seconds - frequency to save restarts
  atmos_model_nml:
    fhout: 0.25 # hours - frequency to save physics outputs
  diag_manager_nml:
    flush_nc_files: true
  fv_core_nml:
    do_sat_adj: false
    warm_start: true
    external_ic: false
    external_eta: false
    nggps_ic: false
    make_nh: false
    mountain: true
    nwat: 2
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
  gfs_physics_nml:
    fhzero: 0.25 # hours - frequency at which precip is set back to zero
    satmedmf: false
    hybedmf: true
    imp_physics: 99
    ncld: 1
diagnostics:
- tensorboard: true
  times:
    kind: every
  chunks:
    time: 8
  variables:
  - storage_of_specific_humidity_path_due_to_fv3_physics
  - storage_of_eastward_wind_path_due_to_fv3_physics
  - storage_of_northward_wind_path_due_to_fv3_physics
  - storage_of_air_temperature_path_due_to_fv3_physics
  - storage_of_specific_humidity_path_due_to_emulator
  - storage_of_eastward_wind_path_due_to_emulator
  - storage_of_northward_wind_path_due_to_emulator
  - storage_of_air_temperature_path_due_to_emulator
EOF

ROOT=/data/prognostic-runs
RUN=$ROOT/$(date -Iseconds)
echo "Running in $RUN"

mkdir -p $ROOT

MONTH=08
IC_URL=gs://vcm-ml-experiments/andrep/2021-05-28/spunup-c48-simple-phys-hybrid-edmf

python3 prepare_config.py config.yaml gs://vcm-ml-experiments/andrep/2021-05-28/spunup-c48-simple-phys-hybrid-edmf 20160802.000000  > fv3config.yaml
runfv3 run-native fv3config.yaml "$RUN" sklearn_runfile.py
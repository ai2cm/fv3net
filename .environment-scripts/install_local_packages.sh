#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

local_packages_to_install=( 
  external/vcm/external/mappm 
)
for package  in "${local_packages_to_install[@]}"
do
  pip install --no-deps -e "$package"
done

poetry_packages=( external/runtime external/report external/gallery . 
  external/fv3config 
  external/vcm 
  external/synth 
  external/fv3kube
  external/loaders
  external/diagnostics_utils
  workflows/one_step_diags 
  workflows/fine_res_budget
  workflows/offline_ml_diags
)

for package in "${poetry_packages[@]}"
do
  (
    cd "$package" || exit
    conda develop .
  )
done

#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

local_packages_to_install=( 
  external/fv3fit
  external/fv3gfs-util
  external/vcm
)
for package  in "${local_packages_to_install[@]}"
do
  pip install -c constraints.txt -e "$package"
done

poetry_packages=( 
  external/report
  external/fv3viz
  external/synth
  external/fv3kube
  external/loaders
  external/diagnostics_utils
  workflows/fine_res_budget
  workflows/offline_ml_diags
  workflows/dataflow
)

for package in "${poetry_packages[@]}"
do
  (
    cd "$package" || exit
    conda develop .
  )
done

# needs to be installed after reports
pip install -c constraints.txt --no-deps -e workflows/prognostic_run_diags

#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

local_packages_to_install=()
for package  in ${local_packages_to_install[@]}
do
  pip install --no-deps -e $package
done

poetry_packages=( external/runtime external/report external/gallery . external/fv3config external/vcm external/vcm/external/mappm  workflows/one_step_diags )

for package in ${poetry_packages[@]}
do
  (
    cd $package
    conda develop .
  )
done

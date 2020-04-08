#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

local_packages_to_install=(. external/fv3config external/vcm external/vcm/external/mappm external/runtime )
for package  in ${local_packages_to_install[@]}
do
  pip install --no-deps -e $package
done

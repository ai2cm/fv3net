#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

local_packages_to_install=(. external/fv3config external/vcm external/vcm/external/mappm )
for package  in ${local_packages_to_install[@]}
do
  pip install --no-deps -e $package
done

poetry_packages=( external/runtime )
for package in ${poetry_packages[@]}
do
  (
    cd $package
    rm -f dist/*.whl
    poetry build
    pip install dist/*.whl
  )
done

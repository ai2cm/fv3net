#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

# we want to force a rebuild in case numpy version changes
# this doesn't rebuild automatically when dependencies change version
rm -f "external/vcm/vcm/mappm.*.so"
rm -rf external/vcm/build

local_packages_to_install=(
  external/vcm
  external/artifacts
  external/loaders
  external/fv3fit
)
set -e
for package  in "${local_packages_to_install[@]}"
do
  pip install -c constraints.txt -e "$package"
done
set +e


# need to pip install these to avoid pkg_resources error
pip install -c constraints.txt external/report

poetry_packages=(
  external/fv3viz
  external/synth
  external/fv3kube
  external/report
  workflows/fine_res_budget
  workflows/dataflow
)

for package in "${poetry_packages[@]}"
do
  (
    cd "$package" || exit
    conda develop .
  )
done

# needs to be installed after reports and fv3viz
pip install -c constraints.txt --no-deps -e workflows/diagnostics

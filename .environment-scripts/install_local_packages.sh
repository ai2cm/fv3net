#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

# we want to force a rebuild in case numpy version changes
# this doesn't rebuild automatically when dependencies change version
rm -f "external/vcm/vcm/mappm.*.so"
rm -rf external/vcm/build

set -e
pip install -c constraints.txt -e external/vcm -e external/artifacts -e external/loaders -e external/fv3fit -e external/report
set +e


# need to pip install these to avoid pkg_resources error
pip install -c constraints.txt external/report

poetry_packages=(
  external/fv3viz
  external/synth
  external/fv3kube
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

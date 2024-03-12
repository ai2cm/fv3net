#!/bin/bash

CONDA_ENV=$1

source activate $CONDA_ENV

# we want to force a rebuild in case numpy version changes
# this doesn't rebuild automatically when dependencies change version
rm -f "external/mappm/mappm/mappm.*.so"
rm -rf external/mappm/build

set -e
uv pip install -c constraints.txt \
  -e external/artifacts \
  -e external/loaders \
  -e external/fv3fit \
  -e external/fv3kube \
  -e external/fv3viz \
  -e external/report \
  -e external/radiation \
  -e external/synth \
  -e external/vcm \
  -e external/mappm \
  -e workflows/fine_res_budget \
  -e workflows/dataflow \
  -e workflows/diagnostics \
  -e workflows/prognostic_scream_run \
  -e external/wandb-query \
  -e external/xtorch_harmonics
set +e

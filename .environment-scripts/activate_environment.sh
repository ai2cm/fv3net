#!/bin/bash
set -e

PLATFORM=$1
SCRIPTS=$2
FV3NET_DIR=$3
CONDA_ENV=$4

ACTIVATE_CONDA_ENVIRONMENT=$SCRIPTS/activate_conda_environment.sh
if [ -f $ACTIVATE_CONDA_ENVIRONMENT];
then
    source $ACTIVATE_CONDA_ENVIRONMENT
fi

# these are needed for "click" to work
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Override microphysics emulation
export VAR_META_PATH=$FV3NET_DIR/external/emulation/microphysics_parameter_metadata.yaml
export OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
chmod +x $FV3NET_DIR/projects/microphysics/scripts/*
export PATH=$FV3NET_DIR/projects/microphysics/scripts:${PATH}

export PYTHONPATH=$FV3NET_DIR/workflows/prognostic_c48_run:$FV3NET_DIR/external/fv3fit:$FV3NET_DIR/external/emulation:$FV3NET_DIR/external/vcm:/fv3net/external/artifacts:$FV3NET_DIR/external/loaders:$FV3NET_DIR/external/fv3kube:$FV3NET_DIR/workflows/post_process_run:$FV3NET_DIR/external/radiation:${PYTHONPATH}

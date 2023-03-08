#!/bin/bash
set -e

PLATFORM=$1
FV3NET_DIR=$2
INSTALL_PREFIX=$3
CONDA_ENV=$4

FV3NET_SCRIPTS=$FV3NET_DIR/.environment-scripts
PLATFORM_SCRIPTS=$FV3NET_SCRIPTS/$PLATFORM

ACTIVATE_CONDA_ENVIRONMENT=$PLATFORM_SCRIPTS/activate_conda_environment.sh
if [ -f $ACTIVATE_CONDA_ENVIRONMENT ];
then
    source $ACTIVATE_CONDA_ENVIRONMENT $CONDA_ENV
fi

# these are needed for "click" to work
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Override microphysics emulation
export VAR_META_PATH=$FV3NET_DIR/external/emulation/microphysics_parameter_metadata.yaml
export OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
export PATH=$FV3NET_DIR/projects/microphysics/scripts:${PATH}

# Add fv3net packages to the PYTHONPATH
export PYTHONPATH=$FV3NET_DIR/workflows/prognostic_c48_run:$FV3NET_DIR/external/fv3fit:$FV3NET_DIR/external/emulation:$FV3NET_DIR/external/vcm:/fv3net/external/artifacts:$FV3NET_DIR/external/loaders:$FV3NET_DIR/external/fv3kube:$FV3NET_DIR/workflows/post_process_run:$FV3NET_DIR/external/radiation:${PYTHONPATH}

# Add shared libraries to LD_LIBRARY_PATH
ESMF_LIB=$INSTALL_PREFIX/esmf/lib
FMS_LIB=$FV3NET_DIR/external/fv3gfs-fortran/FMS/libFMS/.libs/
CALLPYFORT_LIB=$INSTALL_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ESMF_LIB:$FMS_LIB:$CALLPYFORT_LIB

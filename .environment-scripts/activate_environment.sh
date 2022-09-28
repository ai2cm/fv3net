#!/bin/bash
set -e

PLATFORM=$1
FV3NET_DIR=$2
INSTALL_PREFIX=$3  # Must be in environment when sourcing the environment_variable.sh script.
CONDA_ENV=$4

FV3NET_SCRIPTS=$FV3NET_DIR/.environment-scripts
PLATFORM_SCRIPTS=$FV3NET_SCRIPTS/$PLATFORM

ACTIVATE_CONDA_ENVIRONMENT=$PLATFORM_SCRIPTS/activate_conda_environment.sh
if [ -f $ACTIVATE_CONDA_ENVIRONMENT ];
then
    source $ACTIVATE_CONDA_ENVIRONMENT $CONDA_ENV
fi

source $FV3NET_SCRIPTS/environment_variables.sh

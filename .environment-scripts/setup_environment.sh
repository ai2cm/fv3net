#!/bin/bash
set -e

INSTALL_TYPE=$1  # Can be one of "all", "base", or "fv3net"
PLATFORM=$2
CLONE_PREFIX=$3
INSTALL_PREFIX=$4
FV3NET_DIR=$5
FMS_DIR=$6  # This could be in fv3net if we wanted
FV3_DIR=$7  # This could be in fv3net if we wanted -- should be optional (not relevant if we are building just the base image)
CALLPYFORT=$8  # Should be '' if not installed -- should be made an option
CONDA_ENV=$9  # Also optional (not needed in prognostic run docker image for instance)

SCRIPTS=$FV3NET_DIR/.environment-scripts
PLATFORM_SCRIPTS=$SCRIPTS/$PLATFORM
NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
ESMF_DIR=$INSTALL_PREFIX/esmf

MODULES_FILE=$FV3_DIR/conf/modules.fv3.$PLATFORM
if [ -f $MODULES_FILE ];
then
    source $MODULES_FILE
fi

mkdir -p $CLONE_PREFIX
mkdir -p $INSTALL_PREFIX

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "base" ];
then
    bash $PLATFORM_SCRIPTS/install_base_software.sh $INSTALL_PREFIX $CONDA_ENV
    bash $PLATFORM_SCRIPTS/install_nceplibs.sh $SCRIPTS $CLONE_PREFIX/NCEPlibs $NCEPLIBS_DIR
    bash $PLATFORM_SCRIPTS/install_esmf.sh $SCRIPTS $CLONE_PREFIX/esmf $ESMF_DIR
    bash $PLATFORM_SCRIPTS/install_fms.sh $SCRIPTS $FMS_DIR
    if [ ! -z "${CALLPYFORT}" ];
    then
        bash $PLATFORM_SCRIPTS/install_call_py_fort.sh $SCRIPTS $CLONE_PREFIX/call_py_fort
    fi
fi

bash $SCRIPTS/setup_environment_post_base.sh \
    $INSTALL_TYPE \
    $PLATFORM \
    $CLONE_PREFIX \
    $INSTALL_PREFIX \
    $FV3NET_DIR \
    $FMS_DIR \
    $FV3_DIR \
    $CALLPYFORT \
    $CONDA_ENV

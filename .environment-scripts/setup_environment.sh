#!/bin/bash

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

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "fv3net" ];
then
    export NCEPLIBS_DIR=$NCEPLIBS_DIR
    export ESMF_DIR=$ESMF_DIR
    export FMS_DIR=$FMS_DIR
    export FV3_DIR=$FV3_DIR
    if [ ! -z "${CALLPYFORT}" ];
    then
        export CALL_PY_FORT_DIR=$CLONE_PREFIX/call_py_fort
    fi

    ACTIVATE_CONDA=$PLATFORM_SCRIPTS/activate_conda_environment.sh
    if [ -f $ACTIVATE_CONDA ] && [ $INSTALL_TYPE == "fv3net" ];
    then
        source $ACTIVATE_CONDA $CONDA_ENV
    fi

    if [ $PLATFORM != "gnu_docker" ];
    then
        # See fv3net#2046 for more information regarding why we cannot simply use the make
        # rule for this.
        cp $FV3NET_DIR/constraints.txt $FV3NET_DIR/docker/prognostic_run/requirements.txt
    fi
    CALLPYFORT=$CALLPYFORT bash $PLATFORM_SCRIPTS/install_fv3gfs_fortran.sh $SCRIPTS $FV3_DIR $INSTALL_PREFIX
    bash $SCRIPTS/install_fv3net_python_dependencies.sh \
        $FV3NET_DIR/docker/prognostic_run/requirements.txt \
        $FV3NET_DIR/external/vcm \
        $FV3NET_DIR/external/artifacts \
        $FV3NET_DIR/external/loaders \
        $FV3NET_DIR/external/fv3fit \
        $FV3NET_DIR/external/fv3kube \
        $FV3NET_DIR/workflows/post_process_run \
        $FV3NET_DIR/workflows/prognostic_c48_run \
        $FV3NET_DIR/external/emulation \
        $FV3NET_DIR/external/radiation
    CALLPYFORT=$CALLPYFORT bash $PLATFORM_SCRIPTS/install_python_wrapper.sh $SCRIPTS $FV3_DIR
    bash $SCRIPTS/post_build_steps.sh $FV3NET_DIR
fi

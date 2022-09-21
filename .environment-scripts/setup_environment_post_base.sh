#!/bin/bash
set -e

INSTALL_TYPE=$1  # Can be one of "all", "base", "fv3gfs-fortran", "fv3net-python", "wrapper", or "post-build"
PLATFORM=$2
CLONE_PREFIX=$3
INSTALL_PREFIX=$4
FV3NET_DIR=$5
CALLPYFORT=$6  # Should be '' if not installed -- should be made an option
CONDA_ENV=$7  # Also optional (not needed in prognostic run docker image for instance)

FMS_DIR=$FV3NET_DIR/external/fv3gfs-fortran/FMS
FV3_DIR=$FV3NET_DIR/external/fv3gfs-fortran/FV3

SCRIPTS=$FV3NET_DIR/.environment-scripts
PLATFORM_SCRIPTS=$SCRIPTS/$PLATFORM
NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
ESMF_DIR=$INSTALL_PREFIX/esmf

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "fv3gfs-fortran" ] || [ $INSTALL_TYPE == "wrapper" ];
then
    if [ $PLATFORM == "gnu_docker" ];
    then
        export NCEPLIBS_DIR=$NCEPLIBS_DIR/lib
    else
        # I prefer this version, but it is hard-coded in the configuration file for
        # gnu_docker in the fortran repo.
        export NCEPLIBS_DIR=$NCEPLIBS_DIR
    fi
    export ESMF_DIR=$ESMF_DIR
    export FMS_DIR=$FMS_DIR
    export FV3_DIR=$FV3_DIR
    if [ ! -z "${CALLPYFORT}" ];
    then
        export CALL_PY_FORT_DIR=$CLONE_PREFIX/call_py_fort
    fi
fi

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "fv3gfs-fortran" ];
then
    CALLPYFORT=$CALLPYFORT bash $PLATFORM_SCRIPTS/install_fv3gfs_fortran.sh $SCRIPTS $FV3_DIR $INSTALL_PREFIX
fi

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "python-requirements" ] || [ $INSTALL_TYPE == "fv3net-packages" ] || [ $INSTALL_TYPE == "wrapper" ];
then
    ACTIVATE_CONDA=$PLATFORM_SCRIPTS/activate_conda_environment.sh
    if [ -f $ACTIVATE_CONDA ] && [ $INSTALL_TYPE != "all" ];
    then
        source $ACTIVATE_CONDA $CONDA_ENV
    fi
fi

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "python-requirements" ];
then
    if [ $PLATFORM != "gnu_docker" ];
    then
        # See fv3net#2046 for more information regarding why we cannot simply use the make
        # rule for this.
        cp $FV3NET_DIR/constraints.txt $FV3NET_DIR/docker/prognostic_run/requirements.txt
    fi
    bash $PLATFORM_SCRIPTS/install_python_requirements.sh $FV3NET_DIR/docker/prognostic_run/requirements.txt
fi


if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "wrapper" ];
then
    CALLPYFORT=$CALLPYFORT bash $PLATFORM_SCRIPTS/install_python_wrapper.sh $SCRIPTS $FV3_DIR
fi

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "fv3net-packages" ];
then
    bash $SCRIPTS/install_fv3net_packages.sh \
        $FV3NET_DIR/external/vcm \
        $FV3NET_DIR/external/artifacts \
        $FV3NET_DIR/external/loaders \
        $FV3NET_DIR/external/fv3fit \
        $FV3NET_DIR/external/fv3kube \
        $FV3NET_DIR/workflows/post_process_run \
        $FV3NET_DIR/workflows/prognostic_c48_run \
        $FV3NET_DIR/external/emulation \
        $FV3NET_DIR/external/radiation
fi

if [ $INSTALL_TYPE == "all" ] || [ $INSTALL_TYPE == "post-build" ];
then
    bash $SCRIPTS/post_build_steps.sh $FV3NET_DIR
fi

#!/bin/bash
set -e

INSTALL_TYPE=$1  # Can be one of "all", "base", "fv3gfs-fortran", "python-requirements", "wrapper", "fv3net-packages", or "post-build"
PLATFORM=$2
CLONE_PREFIX=$3
INSTALL_PREFIX=$4
FV3NET_DIR=$5
CALLPYFORT=$6  # Should be '' if not installed
CONDA_ENV=$7  # Optional (not needed in prognostic run docker image)

FMS_DIR=$FV3NET_DIR/external/fv3gfs-fortran/FMS
FV3_DIR=$FV3NET_DIR/external/fv3gfs-fortran/FV3

SCRIPTS=$FV3NET_DIR/.environment-scripts
PLATFORM_SCRIPTS=$SCRIPTS/$PLATFORM
NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
ESMF_DIR=$INSTALL_PREFIX/esmf

MODULES_FILE=$FV3_DIR/conf/modules.fv3.$PLATFORM
if [ -f "$MODULES_FILE" ];
then
    source "$MODULES_FILE"
fi

mkdir -p "$CLONE_PREFIX"
mkdir -p "$INSTALL_PREFIX"

source $PLATFORM_SCRIPTS/variables.sh

if [ "$INSTALL_TYPE" == "all" ] || [ "$INSTALL_TYPE" == "base" ];
then
    bash "$PLATFORM_SCRIPTS"/install_base_software.sh "$INSTALL_PREFIX" "$CONDA_ENV"
    bash "$SCRIPTS"/install_nceplibs.sh "$CLONE_PREFIX"/NCEPlibs "$NCEPLIBS_DIR" "$NCEPLIBS_PLATFORM" "$NCEPLIBS_COMPILER"
    bash "$SCRIPTS"/install_esmf.sh "$CLONE_PREFIX"/esmf "$ESMF_DIR" "$ESMF_OS" "$ESMF_COMPILER" "$ESMF_SITE"

    CC="$FMS_CC" \
    FC="$FMS_FC" \
    LDFLAGS="$FMS_LDFLAGS" \
    LOG_DRIVER_FLAGS="$FMS_LOG_DRIVER_FLAGS" \
    CPPFLAGS="$FMS_CPPFLAGS" \
    FCFLAGS="$FMS_FCFLAGS" \
    bash "$SCRIPTS"/install_fms.sh "$FMS_DIR"

    if [ -n "${CALLPYFORT}" ];
    then
        bash "$SCRIPTS"/install_call_py_fort.sh "$CLONE_PREFIX"/call_py_fort
    fi
fi

if [ "$INSTALL_TYPE" != "base" ];
then
    bash "$SCRIPTS"/setup_environment_post_base.sh \
        "$INSTALL_TYPE" \
        "$PLATFORM" \
        "$CLONE_PREFIX" \
        "$INSTALL_PREFIX" \
        "$FV3NET_DIR" \
        "$CALLPYFORT" \
        "$CONDA_ENV"
fi

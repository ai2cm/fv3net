#!/bin/bash

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
PLATFORM=$3
NCEPLIBS_PLATFORM=$4
NCEPLIBS_COMPILER=$5
ESMF_OS=$6
ESMF_COMPILER=$7
ESMF_SITE=$8
FMS_FLAGS=${9}
FV3GFS_FORTRAN_PLATFORM=${10}

source setup-environment-$PLATFORM.sh $INSTALL_PREFIX
bash install-fv3net-python-dependencies.sh
bash install-nceplibs.sh \
    $CLONE_PREFIX/NCEPlibs \
    $INSTALL_PREFIX/NCEPlibs \
    $NCEPLIBS_PLATFORM \
    $NCEPLIBS_COMPILER
bash install-esmf.sh \
    $CLONE_PREFIX/esmf \
    $INSTALL_PREFIX/esmf \
    $ESMF_OS \
    $ESMF_COMPILER \
    $ESMF_SITE \
    $NETCDF_INCLUDE

# ESMF_DIR needs to be set after installing ESMF, so we might as
# well set the rest of the required FV3GFS environment variables here.
CWD=$(pwd)
export NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
export ESMF_DIR=$INSTALL_PREFIX/esmf
export FV3GFS_FORTRAN_DIR=$CWD/../../external/fv3gfs-fortran
export FMS_DIR=$FV3GFS_FORTRAN_DIR/FMS
export FV3_DIR=$FV3GFS_FORTRAN_DIR/FV3

bash install-fms.sh $FMS_DIR $FMS_FLAGS
bash install-fv3gfs-fortran.sh $FV3GFS_FORTRAN_PLATFORM
bash install-python-wrapper.sh

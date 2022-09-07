#!/bin/bash

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
NCEPLIBS_PLATFORM=$3
NCEPLIBS_COMPILER=$4
ESMF_OS=$5
ESMF_COMPILER=$6
ESMF_SITE=$7
FMS_PLATFORM=$8
FV3_PLATFORM=$9

CWD=$(pwd)

export NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
export ESMF_DIR=$INSTALL_PREFIX/esmf
export FV3GFS_FORTRAN_DIR=$CWD/external/fv3gfs-fortran
export FMS_DIR=$FV3GFS_FORTRAN_DIR/FMS
export FV3_DIR=$FV3GFS_FORTRAN_DIR/FV3

bash install-nceplibs.sh \
    $CLONE_PREFIX \
    $NCEPLIBS_DIR \
    $NCEPLIBS_PLATFORM \
    $NCEPLIBS_COMPILER

bash install-esmf.sh \
    $CLONE_PREFIX \
    $ESMF_DIR \
    $ESMF_OS \
    $ESMF_COMPILER \
    $ESMF_SITE

bash install-fms-$FMS_PLATFORM.sh $FMS_DIR
bash install-fv3gfs-fortran.sh \
    $FV3_DIR \
    $FV3_PLATFORM

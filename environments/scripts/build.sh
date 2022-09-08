#!/bin/bash

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
PLATFORM=$3
NCEPLIBS_PLATFORM=$4
NCEPLIBS_COMPILER=$5
ESMF_OS=$6
ESMF_COMPILER=$7
ESMF_SITE=$8
FMS_PLATFORM=${9}
FV3_PLATFORM=${10}

CWD=$(pwd)
export NCEPLIBS_DIR=$INSTALL_PREFIX/NCEPlibs
export FV3GFS_FORTRAN_DIR=$CWD/../../external/fv3gfs-fortran
export FMS_DIR=$FV3GFS_FORTRAN_DIR/FMS
export FV3_DIR=$FV3GFS_FORTRAN_DIR/FV3

source setup-environment-$PLATFORM.sh $INSTALL_PREFIX

bash install-fv3net-python-dependencies.sh

# bash install-nceplibs.sh \
#     $CLONE_PREFIX \
#     $NCEPLIBS_DIR \
#     $NCEPLIBS_PLATFORM \
#     $NCEPLIBS_COMPILER

# bash install-esmf.sh \
#     $CLONE_PREFIX \
#     $INSTALL_PREFIX \
#     $ESMF_OS \
#     $ESMF_COMPILER \
#     $ESMF_SITE \
#     $NETCDF_INCLUDE

# Needs to be set after installing ESMF
export ESMF_DIR=$INSTALL_PREFIX/esmf

# bash install-fms-$FMS_PLATFORM.sh $FMS_DIR
# bash install-fv3gfs-fortran.sh \
#     $FV3_DIR \
#     $FV3_PLATFORM

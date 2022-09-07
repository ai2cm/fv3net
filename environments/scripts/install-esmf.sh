#!/bin/bash

set -e

CLONE_PREFIX=$1
INSTALL_DIR=$2
ESMF_OS=$3
ESMF_COMPILER=$4
ESMF_SITE=$5
NETCDF_INCLUDE=$6

export ESMF_INSTALL_PREFIX=$INSTALL_DIR
export ESMF_INSTALL_MODDIR=$INSTALL_DIR/include
export ESMF_INSTALL_HEADERDIR=$INSTALL_DIR/include
export ESMF_INSTALL_LIBDIR=$INSTALL_DIR/lib
export ESMF_INSTALL_BINDIR=$INSTALL_DIR/bin
export ESMF_NETCDF_INCLUDE=$NETCDF_INCLUDE
export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
export ESMF_BOPT=O3

# These flags set how the build is configured depending on the platform: 
# https://github.com/esmf-org/esmf/tree/develop/build_config
export ESMF_OS=$ESMF_OS
export ESMF_COMPILER=$ESMF_COMPILER
export ESMF_SITE=$ESMF_SITE

git clone -b ESMF_8_0_0 --depth 1 https://github.com/esmf-org/esmf.git $CLONE_PREFIX/esmf
cd $CLONE_PREFIX/esmf
make lib -j24
make install
make installcheck
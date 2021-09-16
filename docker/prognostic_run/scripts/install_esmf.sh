#!/bin/bash

set -e

PREFIX="$1"

export ESMF_DIR=/esmf
export ESMF_INSTALL_PREFIX=$PREFIX
export ESMF_INSTALL_MODDIR=$PREFIX/include
export ESMF_INSTALL_HEADERDIR=$PREFIX/include
export ESMF_INSTALL_LIBDIR=$PREFIX/lib
export ESMF_INSTALL_BINDIR=$PREFIX/bin
export ESMF_NETCDF_INCLUDE=/usr/include
export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
export ESMF_BOPT=O3

git clone -b ESMF_8_0_0 --depth 1 https://git.code.sf.net/p/esmf/esmf $ESMF_DIR
cd $ESMF_DIR
make lib -j24
make install
make installcheck
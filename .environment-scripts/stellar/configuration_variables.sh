#!/bin/bash
set -e
CLONE_PREFIX=$1

# NCEPlibs arguments
NCEPLIBS_PLATFORM=cheyenne
NCEPLIBS_COMPILER=intel

# ESMF arguments and environment variables
ESMF_OS=Linux
ESMF_COMPILER=intel
ESMF_SITE=default
ESMF_CC=mpicc

# FMS environment variables
FMS_CC=mpicc
FMS_FC=mpif90
FMS_LDFLAGS=
FMS_LOG_DRIVER_FLAGS=
FMS_CPPFLAGS='-Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS'
FMS_FCFLAGS='-FR -i4 -r8'
FMS_MAKE_OPTIONS=

# fv3gfs-fortran arguments
FV3GFS_PLATFORM=stellar

# Python requirements environment variables
MPI4PY_CC=mpicc
MPI4PY_MPICC=mpicc

# Python wrapper environment variables
WRAPPER_CC='mpicc'
WRAPPER_LDSHARED='mpicc -shared'

# 'bat' executable destination. Necesary to build  FMS library
export PATH=$CLONE_PREFIX/bats-core/bin:$PATH

#!/bin/bash

ENVIRONMENT_SCRIPTS_DIR=$1
FMS_DIR=$2

CC=/usr/bin/mpicc \
FC=/usr/bin/mpif90 \
LDFLAGS='-L/usr/lib' \
LOG_DRIVER_FLAGS='--comments' \
CPPFLAGS='-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS' \
FCFLAGS='-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp' \
bash $ENVIRONMENT_SCRIPTS_DIR/install_fms.sh $FMS_DIR

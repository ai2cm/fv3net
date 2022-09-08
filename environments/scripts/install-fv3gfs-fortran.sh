#!/bin/bash

FV3GFS_FORTRAN_PLATFORM=$1

cd ../../external/fv3gfs-fortran/FV3
./configure $FV3GFS_FORTRAN_PLATFORM
make -j 8

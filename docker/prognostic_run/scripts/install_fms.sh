#!/bin/bash

set -e

src="$1"

export CC=/usr/bin/mpicc
export FC=/usr/bin/mpif90
export LDFLAGS="-L/usr/lib"
export LOG_DRIVER_FLAGS="--comments"
export CPPFLAGS="-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS"
export FCFLAGS="-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp"

cd $src
autoreconf --install
./configure
make -j8
mv $src/*/*.mod $src/*/*.o $src/*/*.h $src/
